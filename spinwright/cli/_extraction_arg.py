"""Shared resolver for ``--extraction`` across measure/optimize/run.

Extractions always live at ``<workspace>/repo/<corpus_dir>/<name>.py`` — the
only place ``extract`` ever writes them. The CLI arg is therefore a *name*
(the sanitized stem), not a path. We accept three input shapes for ergonomics:

  - bare stem:        `tests_test_x__test_y`
  - stem + ext:       `tests_test_x__test_y.py`
  - path-like:        `spinwright/tests_test_x__test_y.py`  (last component used)

All three resolve to ``<workspace>/repo/<corpus_dir>/tests_test_x__test_y.py``.
The path-like form lets the hint output (which historically printed the
repo-relative path) stay paste-able.
"""

from __future__ import annotations

from pathlib import Path


def resolve_extraction(
    workspace_root: Path, extraction_arg: str, *, corpus_dir: str,
) -> Path:
    """Resolve an --extraction arg to an absolute path inside the corpus dir.
    Raises ``SystemExit`` with a helpful message if the file isn't there."""
    # Peel off any directory prefix; keep only the basename so all input forms
    # collapse to the same stem.
    name = Path(extraction_arg).name
    if not name.endswith(".py"):
        name = f"{name}.py"
    abs_path = (workspace_root / "repo" / corpus_dir / name).resolve()
    if not abs_path.exists():
        raise SystemExit(
            f"extraction {extraction_arg!r} not found at {abs_path}.\n"
            f"Looked in: <workspace>/repo/{corpus_dir}/\n"
            "Did you run `spinwright extract` first?"
        )
    return abs_path


def to_stem(extraction_arg: str) -> str:
    """Render an --extraction arg as a bare stem, for use in printed hints."""
    return Path(extraction_arg).name.removesuffix(".py")
