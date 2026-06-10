from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from spinwright import config as cfg_mod
from spinwright.extraction import extract
from spinwright.repo.workspace import Workspace


# ---------------------------------------------------------------------------
# Fake SDK (same shape as test_dispatch.py — kept local to avoid import churn)
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


class FakeClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


# ---------------------------------------------------------------------------
# Workspace fixture: tmp dir, fake venv (symlink to sys.executable), git init
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path, test_source: str) -> Workspace:
    root = tmp_path / "ws"
    repo = root / "repo"
    venv_dir = root / ".venv"

    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").symlink_to(Path(sys.executable))

    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_mod.py").write_text(
        textwrap.dedent(test_source).lstrip("\n")
    )
    # Trivial package so the extraction can `from target_pkg import ...`
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
        venv_dir=venv_dir,
        branch="spinwright/test",
        base_sha=base_sha,
        keep=True,
    )


_DEFAULT_TEST = """
    from target_pkg import sum_of_squares

    def test_sum_of_squares():
        xs = list(range(100))
        assert sum_of_squares(xs) == sum(x * x for x in xs)
"""


def _config(corpus_dir: str = "extractions") -> cfg_mod.Config:
    return cfg_mod.from_dict(
        {
            "corpus": {"dir": corpus_dir},
            "budget": {"max_extraction_turns": 6},
        }
    )


# ---------------------------------------------------------------------------
# sanitize_test_id
# ---------------------------------------------------------------------------


def test_sanitize_function_id():
    assert (
        extract.sanitize_test_id("tests/test_foo.py::test_bar")
        == "tests_test_foo__test_bar"
    )


def test_sanitize_method_id():
    assert (
        extract.sanitize_test_id("tests/test_foo.py::TestThing::test_bar")
        == "tests_test_foo__TestThing__test_bar"
    )


def test_sanitize_strips_parametrize_suffix():
    assert (
        extract.sanitize_test_id("tests/test_foo.py::test_bar[1]")
        == "tests_test_foo__test_bar"
    )


# ---------------------------------------------------------------------------
# Happy path: LLM writes a good extraction and ends the turn
# ---------------------------------------------------------------------------


_GOOD_EXTRACTION = """
from target_pkg import sum_of_squares

def setup():
    return {"xs": list(range(100))}

def run(state):
    state["_out"] = sum_of_squares(state["xs"])

def verify(state):
    assert state["_out"] == sum(x * x for x in state["xs"])
"""


def test_happy_path_writes_commits_and_succeeds(tmp_path: Path):
    ws = _make_workspace(tmp_path, _DEFAULT_TEST)
    cfg = _config()

    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[
                FakeText(text="writing extraction"),
                FakeToolUse(
                    id="tu_1",
                    name="write_file",
                    input={
                        # The path the orchestrator dictates is the resolved
                        # corpus path — the LLM is supposed to use it. We
                        # mirror that here by writing into corpus_dir.
                        "path": str(
                            (
                                ws.repo_dir
                                / "extractions"
                                / "tests_test_mod__test_sum_of_squares.py"
                            ).resolve()
                        ),
                        "content": _GOOD_EXTRACTION.lstrip("\n"),
                    },
                ),
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeText(text="done")], stop_reason="end_turn"),
    )

    result = extract.extract(
        ws=ws,
        nodeid="tests/test_mod.py::test_sum_of_squares",
        config=cfg,
        client=client,
    )

    assert result.success, result.failure_reason
    assert result.extraction_path is not None
    assert result.extraction_path.exists()
    assert result.notes_path is not None and result.notes_path.exists()
    assert result.commit_sha is not None
    assert result.sanity_passed
    # NOTES.md contains commit SHA + nodeid + date
    notes = result.notes_path.read_text()
    assert "test_sum_of_squares" in notes
    assert ws.base_sha in notes


# ---------------------------------------------------------------------------
# Broken extraction → sanity check fails → no commit
# ---------------------------------------------------------------------------


_BROKEN_EXTRACTION = """
from target_pkg import sum_of_squares

def setup():
    return {"xs": list(range(100))}

def run(state):
    state["_out"] = sum_of_squares(state["xs"])

def verify(state):
    assert state["_out"] == 0   # wrong on purpose
"""


def test_sanity_failure_does_not_commit(tmp_path: Path):
    ws = _make_workspace(tmp_path, _DEFAULT_TEST)
    cfg = _config()

    target = str(
        (
            ws.repo_dir / "extractions" / "tests_test_mod__test_sum_of_squares.py"
        ).resolve()
    )
    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[
                FakeToolUse(
                    id="tu_1",
                    name="write_file",
                    input={"path": target, "content": _BROKEN_EXTRACTION.lstrip("\n")},
                )
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeText(text="should be good")], stop_reason="end_turn"),
    )

    result = extract.extract(
        ws=ws,
        nodeid="tests/test_mod.py::test_sum_of_squares",
        config=cfg,
        client=client,
    )
    assert not result.success
    assert result.sanity_passed is False
    assert result.sanity_error is not None
    assert "AssertionError" in result.sanity_error
    assert result.commit_sha is None
    # Working branch should have no new commits beyond init
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
    assert len(log) == 1


# ---------------------------------------------------------------------------
# LLM never writes the file → orchestrator reports failure
# ---------------------------------------------------------------------------


def test_missing_extraction_reports_failure(tmp_path: Path):
    ws = _make_workspace(tmp_path, _DEFAULT_TEST)
    cfg = _config()
    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[FakeText(text="I refuse to write the file")],
            stop_reason="end_turn",
        ),
    )
    result = extract.extract(
        ws=ws,
        nodeid="tests/test_mod.py::test_sum_of_squares",
        config=cfg,
        client=client,
    )
    assert not result.success
    assert result.extraction_path is None
    assert result.commit_sha is None
    assert "without writing" in (result.failure_reason or "")


# ---------------------------------------------------------------------------
# Ineligible test → bail out before calling the LLM at all
# ---------------------------------------------------------------------------


_INELIGIBLE_TEST = """
    import pytest

    @pytest.mark.parametrize('n', [1, 2])
    def test_param(n):
        assert n > 0
"""


def test_ineligible_test_short_circuits(tmp_path: Path):
    ws = _make_workspace(tmp_path, _INELIGIBLE_TEST)
    cfg = _config()
    client = (
        FakeClient()
    )  # no responses queued — if extract calls the LLM, AssertionError

    result = extract.extract(
        ws=ws,
        nodeid="tests/test_mod.py::test_param",
        config=cfg,
        client=client,
    )
    assert not result.success
    assert result.failure_reason and "ineligible" in result.failure_reason
    assert result.eligibility_reasons
    # FakeClient was never called
    assert client.messages.calls == []


# ---------------------------------------------------------------------------
# Test file missing → fail before LLM
# ---------------------------------------------------------------------------


def test_missing_test_file_short_circuits(tmp_path: Path):
    ws = _make_workspace(tmp_path, _DEFAULT_TEST)
    cfg = _config()
    client = FakeClient()
    result = extract.extract(
        ws=ws,
        nodeid="tests/does_not_exist.py::test_x",
        config=cfg,
        client=client,
    )
    assert not result.success
    assert "not found" in (result.failure_reason or "")
    assert client.messages.calls == []
