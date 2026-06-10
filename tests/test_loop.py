from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinwright import config as cfg_mod
from spinwright.optimization import loop, optimize
from spinwright.profiling.cprofile import CProfileResult, ProfileEntry
from spinwright.repo.workspace import Workspace


# ---------------------------------------------------------------------------
# Fake SDK (compact)
# ---------------------------------------------------------------------------


@dataclass
class FakeText:
    text: str
    type: str = "text"

    def model_dump(self, *, exclude_none=True):
        return {"type": "text", "text": self.text}


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def model_dump(self, *, exclude_none=True):
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

    def model_dump(self, *, exclude_none=True):
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
    def __init__(self):
        self.responses, self.calls = [], []

    def queue(self, *r):
        self.responses.extend(r)

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        if not self.responses:
            raise AssertionError("no fake response queued")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


# ---------------------------------------------------------------------------
# Focus selection — uses fake CProfileResult, no workspace needed
# ---------------------------------------------------------------------------


def _entry(
    funcname: str, filename: str, lineno: int = 1, cumtime: float = 0.0
) -> ProfileEntry:
    return ProfileEntry(
        funcname=funcname,
        filename=filename,
        lineno=lineno,
        calls=1,
        primitive_calls=1,
        tottime=0.0,
        cumtime=cumtime,
        tottime_per_call=0.0,
        cumtime_per_call=0.0,
    )


def _profile(*entries: ProfileEntry) -> CProfileResult:
    return CProfileResult(
        iterations=1,
        total_seconds=0.0,
        entries=tuple(entries),
        verify_passed=True,
        verify_error=None,
    )


def test_select_focus_picks_top_cumtime_in_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    prof = _profile(
        _entry("low", str(repo / "a.py"), cumtime=0.1),
        _entry("hot", str(repo / "b.py"), cumtime=1.0),
        _entry("mid", str(repo / "c.py"), cumtime=0.5),
    )
    focus = loop._select_focus(prof, repo, explored=[])
    assert focus is not None
    assert focus.funcname == "hot"


def test_select_focus_skips_explored(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    hot_entry = _entry("hot", str(repo / "b.py"), lineno=10, cumtime=1.0)
    mid_entry = _entry("mid", str(repo / "c.py"), lineno=20, cumtime=0.5)
    prof = _profile(hot_entry, mid_entry)
    explored_key = loop._explored_key(
        hot_entry.filename, hot_entry.lineno, hot_entry.funcname
    )
    focus = loop._select_focus(prof, repo, explored=[explored_key])
    assert focus is not None
    assert focus.funcname == "mid"


def test_select_focus_skips_files_outside_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    prof = _profile(
        _entry("stdlib_func", "/usr/lib/python3.14/json/__init__.py", cumtime=99.0),
        _entry("user_func", str(repo / "a.py"), cumtime=0.1),
    )
    focus = loop._select_focus(prof, repo, explored=[])
    assert focus is not None
    assert focus.funcname == "user_func"


def test_select_focus_skips_extraction_harness(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "spinwright").mkdir(parents=True)
    prof = _profile(
        _entry("harness", str(repo / "spinwright" / "x.py"), cumtime=99.0),
        _entry("user_func", str(repo / "pkg" / "mod.py"), cumtime=0.1),
    )
    focus = loop._select_focus(prof, repo, explored=[])
    assert focus is not None
    assert focus.funcname == "user_func"


def test_select_focus_returns_none_when_all_explored(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    e1 = _entry("a", str(repo / "x.py"), lineno=1, cumtime=1.0)
    e2 = _entry("b", str(repo / "y.py"), lineno=2, cumtime=0.5)
    prof = _profile(e1, e2)
    explored = [
        loop._explored_key(e1.filename, e1.lineno, e1.funcname),
        loop._explored_key(e2.filename, e2.lineno, e2.funcname),
    ]
    assert loop._select_focus(prof, repo, explored=explored) is None


def test_select_focus_skips_internal_funcnames(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    prof = _profile(
        _entry("<module>", str(repo / "x.py"), cumtime=10.0),
        _entry("<genexpr>", str(repo / "y.py"), cumtime=5.0),
        _entry("real_work", str(repo / "z.py"), cumtime=1.0),
    )
    focus = loop._select_focus(prof, repo, explored=[])
    assert focus is not None
    assert focus.funcname == "real_work"


# ---------------------------------------------------------------------------
# qualname guessing
# ---------------------------------------------------------------------------


def test_guess_qualname_for_module(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    q = loop._guess_qualname(
        str(repo / "static_frame" / "core" / "index.py"),
        "_loc_to_iloc",
        repo,
    )
    assert q == "static_frame.core.index._loc_to_iloc"


def test_guess_qualname_drops_init_suffix(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    q = loop._guess_qualname(
        str(repo / "pkg" / "__init__.py"),
        "helper",
        repo,
    )
    assert q == "pkg.helper"


def test_guess_qualname_returns_none_outside_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    q = loop._guess_qualname(
        str(elsewhere / "mod.py"),
        "f",
        repo,
    )
    assert q is None


# ---------------------------------------------------------------------------
# End-to-end loop test with a real workspace + fake LLM
# ---------------------------------------------------------------------------


_TARGET_PKG = """
import time

def slow_a(n):
    time.sleep(0.003)
    return sum(i for i in range(n))

def slow_b(n):
    time.sleep(0.003)
    return sum(i * i for i in range(n))

def caller(n):
    return slow_a(n) + slow_b(n)
"""


_EXTRACTION = """
from target_pkg import caller

def setup():
    return {"n": 100}

def run(state):
    state["_out"] = caller(state["n"])

def verify(state):
    expected = sum(range(state["n"])) + sum(i * i for i in range(state["n"]))
    assert state["_out"] == expected
"""


def _make_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    root = tmp_path / "ws"
    repo = root / "repo"
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(Path(sys.executable))
    (repo / "target_pkg").mkdir(parents=True)
    (repo / "target_pkg" / "__init__.py").write_text(
        textwrap.dedent(_TARGET_PKG).lstrip("\n")
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


def _cfg(threshold=0.20, repeats=3, max_iters=3) -> cfg_mod.Config:
    return cfg_mod.from_dict(
        {
            "measurement": {
                "improvement_threshold": threshold,
                "walltime_repeats": repeats,
            },
            "budget": {"max_patches_proposed": max_iters, "max_extraction_turns": 4},
        }
    )


def test_loop_accepts_two_separate_focus_optimizations(tmp_path: Path):
    """Two slow functions; LLM removes the sleep in each across two iterations."""
    ws, ext_path = _make_workspace(tmp_path)
    cfg = _cfg(max_iters=3)
    target_file = str((ws.repo_dir / "target_pkg" / "__init__.py").resolve())

    client = FakeClient()
    # iter 1: edit slow_a's sleep
    client.messages.queue(
        FakeMessage(
            content=[
                FakeToolUse(
                    id="tu_a",
                    name="edit_file",
                    input={
                        "path": target_file,
                        "old_string": "def slow_a(n):\n    time.sleep(0.003)\n    ",
                        "new_string": "def slow_a(n):\n    ",
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(
            content=[FakeText(text="killed slow_a sleep")], stop_reason="end_turn"
        ),
    )
    # iter 2: edit slow_b's sleep
    client.messages.queue(
        FakeMessage(
            content=[
                FakeToolUse(
                    id="tu_b",
                    name="edit_file",
                    input={
                        "path": target_file,
                        "old_string": "def slow_b(n):\n    time.sleep(0.003)\n    ",
                        "new_string": "def slow_b(n):\n    ",
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(
            content=[FakeText(text="killed slow_b sleep")], stop_reason="end_turn"
        ),
    )
    # iter 3: no more improvements; LLM ends without editing
    client.messages.queue(
        FakeMessage(content=[FakeText(text="nothing more")], stop_reason="end_turn"),
    )

    result = loop.run_loop(ws=ws, extraction_path=ext_path, config=cfg, client=client)

    assert result.success
    assert result.accepted_count == 2
    assert result.stop_reason in {"max_iterations", "no_remaining_bottlenecks"}
    # Both edits made it to commits
    log = (
        subprocess.run(
            ["git", "-C", str(ws.repo_dir), "log", "--oneline"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    # init + 2 spinwright commits
    assert len(log) >= 3
    spinwright_commits = [l for l in log if "spinwright: optimize" in l]
    assert len(spinwright_commits) == 2


def test_loop_records_explored_for_rejections(tmp_path: Path):
    """LLM declines to edit each iteration; orchestrator should keep moving
    to the next focus rather than re-asking the same one."""
    ws, ext_path = _make_workspace(tmp_path)
    cfg = _cfg(max_iters=3)

    client = FakeClient()
    # All three iterations: LLM ends immediately with no edits.
    for i in range(3):
        client.messages.queue(
            FakeMessage(
                content=[FakeText(text=f"no improvement {i}")], stop_reason="end_turn"
            ),
        )

    result = loop.run_loop(ws=ws, extraction_path=ext_path, config=cfg, client=client)
    assert result.success
    assert result.accepted_count == 0
    # All three iterations added to explored, each with a distinct focus key
    assert len(result.explored) == 3
    assert len(set(result.explored)) == 3
