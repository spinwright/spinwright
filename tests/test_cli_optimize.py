from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinwright.cli import optimize as cli_optimize
from spinwright.repo.workspace import Workspace


# Re-use the same fake-SDK shape used by other CLI tests.


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
        usage = fake.usage.model_dump() if hasattr(fake.usage, "model_dump") else (fake.usage or {})
        return ProviderResponse(content=content, stop_reason=fake.stop_reason, usage=usage)


FakeClient = FakeProvider   # backwards-compat alias for older test bodies


_SLOW = """
import time
def sum_squares(n):
    time.sleep(0.005)
    return sum(i * i for i in range(n))
"""

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
        textwrap.dedent(_SLOW).lstrip("\n")
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


# def test_optimize_cli_accepts_a_real_improvement(tmp_path: Path, capsys):
#     ws, ext_path = _make_workspace(tmp_path)
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
#                         "old_string": "time.sleep(0.005)\n    ",
#                         "new_string": "",
#                     },
#                 )
#             ],
#             stop_reason="tool_use",
#         ),
#         FakeMessage(
#             content=[FakeText(text="removed the sleep")], stop_reason="end_turn"
#         ),
#     )

#     args = argparse.Namespace(
#         workspace=str(ws.root),
#         extraction=str(ext_path),
#         config=None,
#         model=None,
#     )
#     rc = cli_optimize.run(args, provider_factory=lambda _spec: (client, "claude-test"))
#     out = capsys.readouterr().out
#     assert rc == 0
#     assert "ACCEPTED" in out
#     assert "baseline" in out
#     assert "candidate" in out
#     assert "delta:" in out


# def test_optimize_cli_reports_rejection_with_diff(tmp_path: Path, capsys):
#     ws, ext_path = _make_workspace(tmp_path)
#     target_file = str((ws.repo_dir / "target_pkg" / "__init__.py").resolve())

#     client = FakeClient()
#     # LLM makes an edit that DOUBLES the sleep — guaranteed below threshold
#     # regardless of measurement noise. A cosmetic-only edit (e.g. a comment)
#     # turned out to be too noise-sensitive on macOS CI runners where two
#     # back-to-back timeit.autorange measurements can drift 20%+ purely by
#     # chance, occasionally pushing a no-op edit over the gate.
#     client.messages.queue(
#         FakeMessage(
#             content=[
#                 FakeToolUse(
#                     id="tu_1",
#                     name="edit_file",
#                     input={
#                         "path": target_file,
#                         "old_string": "time.sleep(0.005)",
#                         "new_string": "time.sleep(0.010)",
#                     },
#                 )
#             ],
#             stop_reason="tool_use",
#         ),
#         FakeMessage(content=[FakeText(text="oops, slower")], stop_reason="end_turn"),
#     )

#     args = argparse.Namespace(
#         workspace=str(ws.root),
#         extraction=str(ext_path),
#         config=None,
#         model=None,
#     )
#     rc = cli_optimize.run(args, provider_factory=lambda _spec: (client, "claude-test"))
#     out = capsys.readouterr().out
#     assert rc == 1
#     assert "REJECTED" in out
#     assert "threshold" in out
#     assert "attempted diff" in out


# def test_optimize_cli_handles_missing_api_key(tmp_path: Path, capsys):
#     ws, ext_path = _make_workspace(tmp_path)

#     def boom():
#         raise cli_optimize.factory_mod.MissingAPIKeyError("no key")

#     args = argparse.Namespace(
#         workspace=str(ws.root),
#         extraction=str(ext_path),
#         config=None,
#         model=None,
#     )
#     rc = cli_optimize.run(args, provider_factory=boom)
#     assert rc == 2
#     err = capsys.readouterr().err
#     assert "no key" in err
