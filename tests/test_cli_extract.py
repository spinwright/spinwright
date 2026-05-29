from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinwright.cli import extract as cli_extract
from spinwright.repo.workspace import Workspace


# ---------------------------------------------------------------------------
# Fake SDK (shared shape with test_dispatch.py / test_extract.py)
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
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


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


class FakeClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


# ---------------------------------------------------------------------------
# Pre-built workspace under tmp_path (no real venv install needed)
# ---------------------------------------------------------------------------


_DEFAULT_TEST = """
    from target_pkg import sum_of_squares

    def test_sum_of_squares():
        xs = list(range(100))
        assert sum_of_squares(xs) == sum(x * x for x in xs)
"""


def _make_workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    repo = root / "repo"
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(Path(sys.executable))
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_mod.py").write_text(textwrap.dedent(_DEFAULT_TEST).lstrip("\n"))
    (repo / "target_pkg").mkdir()
    (repo / "target_pkg" / "__init__.py").write_text(
        "def sum_of_squares(xs):\n    return sum(x * x for x in xs)\n"
    )
    for cmd in (
        ["git", "init", "-b", "main"],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "add", "."],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "commit", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-b", "spinwright/test"],
        check=True, capture_output=True,
    )
    return Workspace(
        root=root,
        repo_dir=repo,
        venv_dir=venv,
        branch="spinwright/test",
        base_sha=subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        keep=True,
    )


_GOOD_EXTRACTION = """
from target_pkg import sum_of_squares

def setup():
    return {"xs": list(range(100))}

def run(state):
    state["_out"] = sum_of_squares(state["xs"])

def verify(state):
    assert state["_out"] == sum(x * x for x in state["xs"])
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _add_pytest_suite(ws: Workspace) -> None:
    """Add a few tests of varying durations so discovery can find them."""
    (ws.repo_dir / "tests").mkdir(exist_ok=True)
    (ws.repo_dir / "tests" / "test_speeds.py").write_text(textwrap.dedent("""
        import time
        import pytest

        def test_slow_eligible():
            time.sleep(0.2)
            assert sum(i * i for i in range(10)) == 285

        def test_slower_but_ineligible(my_fixture):
            time.sleep(0.3)
            assert my_fixture is not None

        @pytest.fixture
        def my_fixture():
            return object()

        def test_fast():
            assert True
    """).lstrip("\n"))
    subprocess.run(["git", "-C", str(ws.repo_dir), "add", "."],
                   check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(ws.repo_dir),
         "-c", "user.email=x@x", "-c", "user.name=x",
         "commit", "-m", "add tests"],
        check=True, capture_output=True,
    )


def test_extract_cli_lists_candidates_without_extracting(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    # Replace the .venv stub with a real-venv-shaped symlink so pytest is
    # findable on PATH for discovery.
    venv_dir = ws.root / ".venv"
    if venv_dir.is_symlink() or venv_dir.exists():
        if venv_dir.is_symlink() or venv_dir.is_file():
            venv_dir.unlink()
        else:
            import shutil
            shutil.rmtree(venv_dir)
    venv_dir.symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    _add_pytest_suite(ws)

    args = argparse.Namespace(
        repo=str(ws.root),
        test=None,
        list_candidates=True,
        config=None,
    )
    client = FakeClient()  # never called for --list-candidates
    rc = cli_extract.run(args, client_factory=lambda: client)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Eligible candidates" in out
    assert "test_slow_eligible" in out
    assert "Rejected candidates" in out
    assert "test_slower_but_ineligible" in out
    # The LLM wasn't called
    assert client.messages.calls == []


def test_extract_cli_auto_selects_when_test_omitted(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    venv_dir = ws.root / ".venv"
    if venv_dir.is_symlink() or venv_dir.exists():
        if venv_dir.is_symlink() or venv_dir.is_file():
            venv_dir.unlink()
        else:
            import shutil
            shutil.rmtree(venv_dir)
    venv_dir.symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    # Lower the slow threshold so test_slow_eligible qualifies even on a fast
    # box; we want discovery to find it deterministically.
    cfg_path = tmp_path / "spinwright.toml"
    cfg_path.write_text("[test_selection]\nslow_threshold_seconds = 0.05\n")
    _add_pytest_suite(ws)

    # Auto-selection should pick `test_slow_eligible` (slowest eligible).
    # The fake LLM scripts a successful extraction so the run reports success.
    target = str(
        (ws.repo_dir / "test_fixtures" / "spinwright"
         / "tests_test_speeds__test_slow_eligible.py").resolve()
    )
    extraction = (
        "import time\n"
        "def setup(): return {}\n"
        "def run(state): time.sleep(0)\n"
        "def verify(state): assert sum(i * i for i in range(10)) == 285\n"
    )
    client = FakeClient()
    client.messages.queue(
        FakeMessage(content=[FakeToolUse(id="tu_1", name="write_file",
                                          input={"path": target, "content": extraction})],
                    stop_reason="tool_use"),
        FakeMessage(content=[FakeText(text="done")], stop_reason="end_turn"),
    )

    args = argparse.Namespace(
        repo=str(ws.root),
        test=None,
        list_candidates=False,
        config=str(cfg_path),
    )
    rc = cli_extract.run(args, client_factory=lambda: client)
    err = capsys.readouterr().err
    assert "Auto-selected" in err
    assert "test_slow_eligible" in err
    assert rc == 0


def test_extract_cli_reuses_existing_workspace_and_succeeds(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    # Default config uses corpus.dir = "test_fixtures/spinwright"; the CLI test
    # exercises that default path (test_extract.py's narrower tests override it).
    target = str(
        (ws.repo_dir / "test_fixtures" / "spinwright"
         / "tests_test_mod__test_sum_of_squares.py").resolve()
    )

    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[FakeToolUse(id="tu_1", name="write_file",
                                 input={"path": target, "content": _GOOD_EXTRACTION.lstrip("\n")})],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeText(text="done")], stop_reason="end_turn"),
    )

    args = argparse.Namespace(
        repo=str(ws.root),
        test="tests/test_mod.py::test_sum_of_squares",
        list_candidates=False,
        config=None,
    )
    rc = cli_extract.run(args, client_factory=lambda: client)
    assert rc == 0
    captured = capsys.readouterr()
    assert "reusing workspace" in captured.err
    assert "Extraction succeeded" in captured.out
    assert "tests_test_mod__test_sum_of_squares.py" in captured.out
    # Real commit on the working branch
    log = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "log", "--oneline"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(log) == 2
    assert "spinwright: extract" in log[0]


def test_extract_cli_reports_ineligible_test_without_calling_llm(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    # Overwrite the test with a parametrized one (ineligible).
    (ws.repo_dir / "tests" / "test_mod.py").write_text(textwrap.dedent("""
        import pytest

        @pytest.mark.parametrize('n', [1, 2])
        def test_param(n):
            assert n > 0
    """).lstrip("\n"))
    client = FakeClient()  # no responses — extract should never call create()

    args = argparse.Namespace(
        repo=str(ws.root),
        test="tests/test_mod.py::test_param",
        list_candidates=False,
        config=None,
    )
    rc = cli_extract.run(args, client_factory=lambda: client)
    assert rc == 1
    captured = capsys.readouterr()
    assert "Extraction FAILED" in captured.out
    assert "ineligible" in captured.out
    assert "pytest_marker" in captured.out
    assert client.messages.calls == []


def test_extract_cli_handles_missing_api_key(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)

    def boom():
        raise cli_extract.client_mod.MissingAPIKeyError("no key")

    args = argparse.Namespace(
        repo=str(ws.root),
        test="tests/test_mod.py::test_sum_of_squares",
        list_candidates=False,
        config=None,
    )
    rc = cli_extract.run(args, client_factory=boom)
    assert rc == 2
    captured = capsys.readouterr()
    assert "no key" in captured.err
