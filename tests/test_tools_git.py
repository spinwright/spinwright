from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spinwright.tools import edit, git


def _init_repo(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "ws"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "src.py").write_text("def hello():\n    return 'world'\n")
    for cmd in (
        ["git", "init", "-b", "main"],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "add", "."],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "commit", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    return workspace, repo


def test_git_diff_is_empty_on_clean_tree(tmp_path: Path):
    workspace, repo = _init_repo(tmp_path)
    assert git.git_diff(repo) == ""


def test_git_diff_shows_uncommitted_changes(tmp_path: Path):
    workspace, repo = _init_repo(tmp_path)
    (repo / "src.py").write_text("def hello():\n    return 'WORLD'\n")
    diff = git.git_diff(repo)
    assert "+    return 'WORLD'" in diff
    assert "-    return 'world'" in diff


def test_git_revert_path_restores_single_file(tmp_path: Path):
    workspace, repo = _init_repo(tmp_path)
    (repo / "src.py").write_text("mutated\n")
    (repo / "other.py").write_text("also mutated\n")  # untracked but exists
    result = git.git_revert_path(workspace, repo, "repo/src.py")
    assert len(result.reverted_paths) == 1
    assert (repo / "src.py").read_text() == "def hello():\n    return 'world'\n"
    # Untracked file unchanged
    assert (repo / "other.py").read_text() == "also mutated\n"


def test_git_revert_all_resets_tracked_files_only(tmp_path: Path):
    workspace, repo = _init_repo(tmp_path)
    (repo / "src.py").write_text("mutated\n")
    (repo / "fresh.py").write_text("not tracked\n")  # untracked
    result = git.git_revert_all(repo)
    assert (repo / "src.py").read_text() == "def hello():\n    return 'world'\n"
    # Untracked survives (so newly-created extractions are preserved).
    assert (repo / "fresh.py").exists()
    # Only the changed-and-tracked file was reported as reverted.
    assert any("src.py" in p for p in result.reverted_paths)
    assert not any("fresh.py" in p for p in result.reverted_paths)


def test_git_revert_all_on_clean_tree_is_noop(tmp_path: Path):
    workspace, repo = _init_repo(tmp_path)
    result = git.git_revert_all(repo)
    assert result.reverted_paths == []


def test_git_revert_path_rejects_workspace_escape(tmp_path: Path):
    workspace, _ = _init_repo(tmp_path)
    with pytest.raises(edit.WorkspaceEscapeError):
        git.git_revert_path(workspace, workspace / "repo", "../escape")


def test_git_revert_path_rejects_inside_workspace_but_outside_repo(tmp_path: Path):
    """Workspace contains repo/, .venv/, etc. A path under .venv/ is inside
    the workspace but not under git's view of the repo."""
    workspace, repo = _init_repo(tmp_path)
    venv_file = workspace / ".venv" / "fake.txt"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("not in git\n")
    with pytest.raises(edit.WorkspaceEscapeError, match="not under the repo dir"):
        git.git_revert_path(workspace, repo, str(venv_file))
