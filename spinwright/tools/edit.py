from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class WorkspaceEscapeError(PermissionError):
    """Raised when a tool-supplied path resolves outside the workspace root."""


class EditMatchError(ValueError):
    """Raised when ``edit_file``'s ``old_string`` doesn't appear exactly once."""


@dataclass(frozen=True)
class WriteResult:
    path: str
    bytes_written: int
    created: bool


@dataclass(frozen=True)
class EditResult:
    path: str
    bytes_written: int


def _resolve_within(workspace_root: Path, raw_path: str) -> Path:
    """Resolve ``raw_path`` to an absolute path under ``workspace_root``.

    Relative paths are joined onto the workspace root. Absolute paths are
    rejected unless they are already under ``workspace_root`` after resolution.
    Symlink-escape and ``..`` traversal are caught by ``.resolve()`` followed
    by a ``relative_to`` containment check.
    """
    root = workspace_root.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise WorkspaceEscapeError(
            f"path {raw_path!r} resolves to {candidate} which is outside workspace {root}"
        )
    return candidate


def write_file(workspace_root: Path, path: str, content: str) -> WriteResult:
    """Create or overwrite ``path`` (inside the workspace) with ``content``."""
    abs_path = _resolve_within(workspace_root, path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    created = not abs_path.exists()
    abs_path.write_text(content)
    return WriteResult(
        path=str(abs_path),
        bytes_written=len(content.encode("utf-8")),
        created=created,
    )


def edit_file(
    workspace_root: Path, path: str, old_string: str, new_string: str
) -> EditResult:
    """Replace ``old_string`` with ``new_string`` in ``path``.

    ``old_string`` must occur exactly once in the file; otherwise
    ``EditMatchError`` is raised. This avoids accidental multi-match edits
    that LLMs occasionally request.
    """
    abs_path = _resolve_within(workspace_root, path)
    if not abs_path.exists():
        raise FileNotFoundError(f"file not found: {abs_path}")
    content = abs_path.read_text()
    occurrences = content.count(old_string)
    if occurrences == 0:
        raise EditMatchError(f"old_string not found in {abs_path}")
    if occurrences > 1:
        raise EditMatchError(
            f"old_string matches {occurrences} locations in {abs_path}; "
            "make it unique by including more surrounding context"
        )
    new_content = content.replace(old_string, new_string, 1)
    abs_path.write_text(new_content)
    return EditResult(
        path=str(abs_path),
        bytes_written=len(new_content.encode("utf-8")),
    )
