from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from spinwright.optimization import regression
from spinwright.repo.workspace import Workspace


PY = Path(sys.executable)


# ---------------------------------------------------------------------------
# Workspace builder: a tiny repo with a pytest suite + a target package, plus
# helpers that commit successive edits to that package.
# ---------------------------------------------------------------------------


_TARGET_INIT = """
def add(a, b):
    return a + b

def mul(a, b):
    return a * b
"""

_SUITE = """
from target_pkg import add, mul

def test_add():
    assert add(2, 3) == 5

def test_mul():
    assert mul(2, 3) == 6
"""


def _make_workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    repo = root / "repo"
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(PY)

    (repo / "target_pkg").mkdir(parents=True)
    (repo / "target_pkg" / "__init__.py").write_text(textwrap.dedent(_TARGET_INIT).lstrip("\n"))
    (repo / "tests").mkdir()
    (repo / "tests" / "test_target.py").write_text(textwrap.dedent(_SUITE).lstrip("\n"))

    for cmd in (
        ["git", "init", "-b", "main"],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "add", "."],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "commit", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    base_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-b", "spinwright/test"],
        check=True, capture_output=True,
    )
    return Workspace(
        root=root, repo_dir=repo, venv_dir=venv,
        branch="spinwright/test", base_sha=base_sha, keep=True,
    )


def _commit_edit(ws: Workspace, path_rel: str, old: str, new: str, msg: str) -> str:
    """Replace `old` with `new` in <repo>/path_rel and commit. Returns the SHA."""
    p = ws.repo_dir / path_rel
    p.write_text(p.read_text().replace(old, new))
    subprocess.run(["git", "-C", str(ws.repo_dir), "add", path_rel],
                   check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(ws.repo_dir),
         "-c", "user.email=x@x", "-c", "user.name=x",
         "commit", "-m", msg],
        check=True, capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(ws.repo_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


# We need pytest installed in the "venv" so the regression check can invoke it.
# The symlinked sys.executable already has pytest available (it's in our
# dev venv), so the subprocess pytest call succeeds without any extra setup.


def test_regression_check_passes_when_suite_green(tmp_path: Path):
    """All accepted commits are functionally fine → no drops."""
    ws = _make_workspace(tmp_path)
    sha1 = _commit_edit(ws, "target_pkg/__init__.py",
                        "def add(a, b):\n    return a + b",
                        "def add(a, b):\n    return (a) + (b)  # cosmetic",
                        "perf: cosmetic")
    sha2 = _commit_edit(ws, "target_pkg/__init__.py",
                        "def mul(a, b):\n    return a * b",
                        "def mul(a, b):\n    return (a) * (b)  # cosmetic",
                        "perf: cosmetic2")
    result = regression.run_regression_check(ws, [sha1, sha2], PY)
    assert result.passed
    assert result.dropped_commits == []
    assert result.fallback_used == "none"


def test_regression_check_drops_single_bad_commit(tmp_path: Path):
    """One of two accepted commits breaks tests. Linear-revert should find and
    drop it; the other survives."""
    ws = _make_workspace(tmp_path)
    sha_good = _commit_edit(ws, "target_pkg/__init__.py",
                            "def add(a, b):\n    return a + b",
                            "def add(a, b):\n    return (a) + (b)  # good",
                            "perf: good edit")
    sha_bad = _commit_edit(ws, "target_pkg/__init__.py",
                           "def mul(a, b):\n    return a * b",
                           "def mul(a, b):\n    return a + b  # WRONG",
                           "perf: bad edit")
    result = regression.run_regression_check(ws, [sha_good, sha_bad], PY)
    assert result.passed
    assert result.dropped_commits == [sha_bad]
    assert result.fallback_used == "linear_revert"
    # Working branch still has the good commit
    log = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "log", "--oneline"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    # init + 1 good cherry-pick
    assert len(log) == 2


def test_regression_check_drops_all_when_no_single_revert_helps(tmp_path: Path):
    """Two commits BOTH break tests individually; no single revert restores
    green. Linear-revert fails, fallback drops all."""
    ws = _make_workspace(tmp_path)
    sha_break1 = _commit_edit(ws, "target_pkg/__init__.py",
                              "def add(a, b):\n    return a + b",
                              "def add(a, b):\n    return a - b  # WRONG",
                              "perf: bad1")
    sha_break2 = _commit_edit(ws, "target_pkg/__init__.py",
                              "def mul(a, b):\n    return a * b",
                              "def mul(a, b):\n    return a + b  # WRONG",
                              "perf: bad2")
    result = regression.run_regression_check(ws, [sha_break1, sha_break2], PY)
    assert result.passed
    assert sorted(result.dropped_commits) == sorted([sha_break1, sha_break2])
    assert result.fallback_used == "drop_all"
    # Working branch should be back at base_sha (only the init commit).
    log = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "log", "--oneline"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(log) == 1


def test_regression_check_finds_culprit_among_three(tmp_path: Path):
    """Three commits; middle one is the culprit. Linear revert should drop it
    in two passes through the loop (depends on iteration order)."""
    ws = _make_workspace(tmp_path)
    sha1 = _commit_edit(ws, "target_pkg/__init__.py",
                        "def add(a, b):\n    return a + b",
                        "def add(a, b):\n    return (a) + (b)",
                        "perf: edit1")
    sha2 = _commit_edit(ws, "target_pkg/__init__.py",
                        "def mul(a, b):\n    return a * b",
                        "def mul(a, b):\n    return 0  # WRONG",
                        "perf: WRONG mul")
    sha3_path = ws.repo_dir / "target_pkg" / "__init__.py"
    sha3_path.write_text(sha3_path.read_text() + "\ndef noop():\n    return None\n")
    subprocess.run(["git", "-C", str(ws.repo_dir), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(ws.repo_dir),
         "-c", "user.email=x@x", "-c", "user.name=x",
         "commit", "-m", "perf: add noop"],
        check=True, capture_output=True,
    )
    sha3 = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    result = regression.run_regression_check(ws, [sha1, sha2, sha3], PY)
    assert result.passed
    assert result.dropped_commits == [sha2]
    assert result.fallback_used == "linear_revert"
