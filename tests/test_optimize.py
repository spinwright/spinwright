from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinwright import config as cfg_mod
from spinwright.measurement.types import WalltimeResult
from spinwright.optimization import optimize
from spinwright.repo.workspace import Workspace


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


@dataclass
class FakeText:
    text: str
    type: str = "text"

    def model_dump(self, *, exclude_none: bool = True) -> dict:
        return {"type": "text", "text": self.text}


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def model_dump(self, *, exclude_none: bool = True) -> dict:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def model_dump(self, *, exclude_none: bool = True) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self) -> None:
        self.responses: list[FakeMessage] = []
        self.calls: list[dict] = []

    def queue(self, *responses: FakeMessage) -> None:
        self.responses.extend(responses)

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("no fake response queued for this call")
        return self.responses.pop(0)


class FakeProvider:
    """Provider-protocol fake: queue FakeMessage objects; create_message pops and
    returns ProviderResponse. messages.calls retains kwargs for assertions."""

    name = "fake"

    def __init__(self) -> None:
        self.messages = FakeMessages()

    def create_message(self, **kwargs):
        from spinwright.llm.providers.base import ProviderResponse

        self.messages.calls.append(copy.deepcopy(kwargs))
        if not self.messages.responses:
            raise AssertionError("no fake response queued for this call")
        fake = self.messages.responses.pop(0)
        content = [b.model_dump() for b in fake.content]
        usage = (
            fake.usage.model_dump()
            if hasattr(fake.usage, "model_dump")
            else (fake.usage or {})
        )
        return ProviderResponse(
            content=content, stop_reason=fake.stop_reason, usage=usage
        )


FakeClient = FakeProvider  # backwards-compat alias for older test bodies


# ---------------------------------------------------------------------------
# Workspace + extraction fixture
# ---------------------------------------------------------------------------


_SLOW_TARGET = """
import time

def sum_squares(n):
    time.sleep(0.005)  # intentionally slow for the optimizer to fix
    return sum(i * i for i in range(n))
"""


_FAST_TARGET_REPLACEMENT = (
    "time.sleep(0.005)  # intentionally slow for the optimizer to fix\n    ",
    "",  # remove the sleep entirely
)


_EXTRACTION = """
from target_pkg import sum_squares

def setup():
    return {"n": 200}

def run(state):
    state["_out"] = sum_squares(state["n"])

def verify(state):
    assert state["_out"] == sum(i * i for i in range(state["n"]))
"""


def _make_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    root = tmp_path / "ws"
    repo = root / "repo"
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(Path(sys.executable))

    (repo / "target_pkg").mkdir(parents=True)
    (repo / "target_pkg" / "__init__.py").write_text(
        textwrap.dedent(_SLOW_TARGET).lstrip("\n")
    )
    extractions = repo / "spinwright"
    extractions.mkdir(parents=True)
    (extractions / "__init__.py").write_text("")
    ext_path = extractions / "demo.py"
    ext_path.write_text(textwrap.dedent(_EXTRACTION).lstrip("\n"))

    for cmd in (
        ["git", "init", "-b", "main"],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "add", "."],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "commit", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    base_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-b", "spinwright/test"],
        check=True,
        capture_output=True,
    )
    return Workspace(
        root=root,
        repo_dir=repo,
        venv_dir=venv,
        branch="spinwright/test",
        base_sha=base_sha,
        keep=True,
    ), ext_path


def _cfg(threshold: float = 0.20, repeats: int = 3) -> cfg_mod.Config:
    return cfg_mod.from_dict(
        {
            "measurement": {
                "improvement_threshold": threshold,
                "walltime_repeats": repeats,
            },
            "budget": {"max_extraction_turns": 6},
        }
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def _wt(median: float) -> WalltimeResult:
    return WalltimeResult(
        best_seconds=median,
        median_seconds=median,
        stddev_seconds=0.0,
        iterations_per_repeat=1,
        repeats=1,
    )


def test_relative_improvement_basic():
    import pytest as _pt

    assert optimize._relative_improvement(_wt(1.0), _wt(0.8)) == _pt.approx(0.2)
    assert optimize._relative_improvement(_wt(1.0), _wt(2.0)) == _pt.approx(-1.0)
    assert optimize._relative_improvement(_wt(0.0), _wt(0.5)) is None


def test_diff_paths_parses_diff_git_lines(tmp_path: Path):
    diff = (
        "diff --git a/foo/bar.py b/foo/bar.py\n"
        "index 1..2 100644\n"
        "--- a/foo/bar.py\n"
        "+++ b/foo/bar.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n+new\n"
    )
    paths = optimize._diff_paths(tmp_path, diff)
    assert paths == [(tmp_path / "foo" / "bar.py").resolve()]


# ---------------------------------------------------------------------------
# Orchestrator: happy path (LLM removes the sleep, big improvement, accepted)
# ---------------------------------------------------------------------------


# def test_accepted_when_improvement_exceeds_threshold(tmp_path: Path):
#     ws, ext_path = _make_workspace(tmp_path)
#     cfg = _cfg(threshold=0.20, repeats=3)
#     target_file = str((ws.repo_dir / "target_pkg" / "__init__.py").resolve())

#     client = FakeClient()
#     client.messages.queue(
#         FakeMessage(
#             content=[
#                 FakeToolUse(
#                     id="tu_1",
#                     name="edit_file",
#                     input={
#                         "path": target_file,
#                         "old_string": _FAST_TARGET_REPLACEMENT[0],
#                         "new_string": _FAST_TARGET_REPLACEMENT[1],
#                     },
#                 )
#             ],
#             stop_reason="tool_use",
#         ),
#         FakeMessage(
#             content=[FakeText(text="removed the sleep")], stop_reason="end_turn"
#         ),
#     )

#     result = optimize.optimize_once(
#         ws=ws,
#         extraction_path=ext_path,
#         config=cfg,
#         provider=client,
#         model="claude-test",
#     )
#     assert result.accepted, result.rejection_reason
#     assert result.commit_sha is not None
#     assert result.relative_improvement is not None
#     assert result.relative_improvement > 0.20
#     assert "time.sleep" in result.diff
#     # The commit lands on the working branch
#     log = (
#         subprocess.run(
#             ["git", "-C", str(ws.repo_dir), "log", "--oneline"],
#             capture_output=True,
#             text=True,
#             check=True,
#         )
#         .stdout.strip()
#         .splitlines()
#     )
#     assert len(log) == 2
#     assert "spinwright: optimize" in log[0]


# ---------------------------------------------------------------------------
# LLM makes no edits → rejected
# ---------------------------------------------------------------------------


# def test_no_edits_rejects_without_revert(tmp_path: Path):
#     ws, ext_path = _make_workspace(tmp_path)
#     cfg = _cfg()
#     client = FakeClient()
#     client.messages.queue(
#         FakeMessage(content=[FakeText(text="nothing to do")], stop_reason="end_turn"),
#     )
#     result = optimize.optimize_once(
#         ws=ws,
#         extraction_path=ext_path,
#         config=cfg,
#         provider=client,
#         model="claude-test",
#     )
#     assert not result.accepted
#     assert "without applying any edits" in (result.rejection_reason or "")
#     assert result.candidate_walltime is None  # never re-measured
#     assert result.commit_sha is None


# ---------------------------------------------------------------------------
# LLM breaks verify → rejected + reverted
# ---------------------------------------------------------------------------


_BROKEN_REPLACEMENT = (
    "return sum(i * i for i in range(n))",
    "return 0  # broken on purpose",
)


# def test_broken_verify_rejects_and_reverts(tmp_path: Path):
#     ws, ext_path = _make_workspace(tmp_path)
#     cfg = _cfg()
#     target_file = str((ws.repo_dir / "target_pkg" / "__init__.py").resolve())

#     client = FakeClient()
#     client.messages.queue(
#         FakeMessage(
#             content=[
#                 FakeToolUse(
#                     id="tu_1",
#                     name="edit_file",
#                     input={
#                         "path": target_file,
#                         "old_string": _BROKEN_REPLACEMENT[0],
#                         "new_string": _BROKEN_REPLACEMENT[1],
#                     },
#                 )
#             ],
#             stop_reason="tool_use",
#         ),
#         FakeMessage(content=[FakeText(text="done")], stop_reason="end_turn"),
#     )

#     original = (ws.repo_dir / "target_pkg" / "__init__.py").read_text()
#     result = optimize.optimize_once(
#         ws=ws,
#         extraction_path=ext_path,
#         config=cfg,
#         provider=client,
#         model="claude-test",
#     )
#     assert not result.accepted
#     assert result.candidate_verify is not None
#     assert result.candidate_verify.passed is False
#     assert "verify" in (result.rejection_reason or "").lower()
#     # File restored to HEAD
#     assert (ws.repo_dir / "target_pkg" / "__init__.py").read_text() == original
#     assert result.reverted_paths


# ---------------------------------------------------------------------------
# LLM makes a slowing edit → rejected for below-threshold + reverted
# ---------------------------------------------------------------------------


# _SLOWER_REPLACEMENT = (
#     "return sum(i * i for i in range(n))",
#     # Add an extra small sleep so candidate is even slower than baseline.
#     "time.sleep(0.003)\n    return sum(i * i for i in range(n))",
# )


# def test_slowing_edit_rejected_and_reverted(tmp_path: Path):
#     ws, ext_path = _make_workspace(tmp_path)
#     cfg = _cfg(threshold=0.20, repeats=3)
#     target_file = str((ws.repo_dir / "target_pkg" / "__init__.py").resolve())

#     client = FakeClient()
#     client.messages.queue(
#         FakeMessage(
#             content=[
#                 FakeToolUse(
#                     id="tu_1",
#                     name="edit_file",
#                     input={
#                         "path": target_file,
#                         "old_string": _SLOWER_REPLACEMENT[0],
#                         "new_string": _SLOWER_REPLACEMENT[1],
#                     },
#                 )
#             ],
#             stop_reason="tool_use",
#         ),
#         FakeMessage(
#             content=[FakeText(text="made it slower oops")], stop_reason="end_turn"
#         ),
#     )

#     original = (ws.repo_dir / "target_pkg" / "__init__.py").read_text()
#     result = optimize.optimize_once(
#         ws=ws,
#         extraction_path=ext_path,
#         config=cfg,
#         provider=client,
#         model="claude-test",
#     )
#     assert not result.accepted
#     assert result.candidate_verify is not None
#     assert (
#         result.candidate_verify.passed is True
#     )  # verify still passes — it's just slower
#     assert result.relative_improvement is not None
#     assert result.relative_improvement < 0.20
#     assert "threshold" in (result.rejection_reason or "")
#     # File restored
#     assert (ws.repo_dir / "target_pkg" / "__init__.py").read_text() == original
