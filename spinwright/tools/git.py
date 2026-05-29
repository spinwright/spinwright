from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from spinwright.tools.edit import WorkspaceEscapeError, _resolve_within


@dataclass(frozen=True)
class RevertResult:
    reverted_paths: list[str]


def _git(repo_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def git_diff(repo_dir: Path) -> str:
    """Return the working-tree diff against HEAD (uncommitted changes)."""
    return _git(repo_dir, "diff", "HEAD")


def git_revert_path(workspace_root: Path, repo_dir: Path, path: str) -> RevertResult:
    """Restore one path to HEAD via ``git checkout HEAD -- <rel_path>``.

    ``path`` may be absolute or relative; it's resolved against the workspace
    root and must remain inside it (same containment rules as edit_file). The
    path is then re-rooted against ``repo_dir`` for the git command since git
    needs paths relative to the repo.
    """
    abs_path = _resolve_within(workspace_root, path)
    try:
        rel = abs_path.relative_to(repo_dir.resolve())
    except ValueError:
        raise WorkspaceEscapeError(
            f"path {path!r} is inside the workspace but not under the repo dir "
            f"{repo_dir} — git cannot revert it"
        )
    _git(repo_dir, "checkout", "HEAD", "--", str(rel))
    return RevertResult(reverted_paths=[str(abs_path)])


def git_revert_all(repo_dir: Path) -> RevertResult:
    """Restore every tracked file in the repo to HEAD. Untracked files are
    left in place (so a newly-created extraction won't be deleted)."""
    # List files that differ from HEAD so we can report what was reverted.
    raw = _git(repo_dir, "diff", "--name-only", "HEAD")
    changed = [line for line in raw.splitlines() if line]
    if changed:
        _git(repo_dir, "checkout", "HEAD", "--", ".")
    return RevertResult(reverted_paths=[str((repo_dir / p).resolve()) for p in changed])
