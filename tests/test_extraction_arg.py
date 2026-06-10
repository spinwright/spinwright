from __future__ import annotations

from pathlib import Path

import pytest

from spinwright.cli._extraction_arg import resolve_extraction, to_stem


def _make_corpus(
    tmp_path: Path, *, corpus_dir: str = "spinwright"
) -> tuple[Path, Path]:
    """Returns (workspace_root, extraction_abs_path)."""
    ws = tmp_path / "ws"
    (ws / "repo" / corpus_dir).mkdir(parents=True)
    ext = ws / "repo" / corpus_dir / "demo_test.py"
    ext.write_text("def setup(): return {}\ndef run(s): pass\ndef verify(s): pass\n")
    return ws, ext


# ---------------------------------------------------------------------------
# resolve_extraction
# ---------------------------------------------------------------------------


def test_bare_stem(tmp_path: Path):
    ws, expected = _make_corpus(tmp_path)
    assert (
        resolve_extraction(ws, "demo_test", corpus_dir="spinwright")
        == expected.resolve()
    )


def test_stem_with_py_extension(tmp_path: Path):
    ws, expected = _make_corpus(tmp_path)
    assert (
        resolve_extraction(ws, "demo_test.py", corpus_dir="spinwright")
        == expected.resolve()
    )


def test_repo_relative_path_form(tmp_path: Path):
    ws, expected = _make_corpus(tmp_path)
    assert (
        resolve_extraction(
            ws,
            "spinwright/demo_test.py",
            corpus_dir="spinwright",
        )
        == expected.resolve()
    )


def test_absolute_path_form_uses_basename(tmp_path: Path):
    """Even if the user pastes an absolute path, only the basename matters —
    the file must still live in the corpus dir under the workspace."""
    ws, expected = _make_corpus(tmp_path)
    assert (
        resolve_extraction(
            ws,
            str(expected.resolve()),
            corpus_dir="spinwright",
        )
        == expected.resolve()
    )


def test_unknown_extraction_errors_with_hint(tmp_path: Path):
    ws, _ = _make_corpus(tmp_path)
    with pytest.raises(SystemExit) as ei:
        resolve_extraction(ws, "no_such_thing", corpus_dir="spinwright")
    msg = str(ei.value)
    assert "no_such_thing" in msg
    assert "spinwright extract" in msg


def test_custom_corpus_dir(tmp_path: Path):
    ws, expected = _make_corpus(tmp_path, corpus_dir="static_frame/test/spinwright")
    assert (
        resolve_extraction(
            ws,
            "demo_test",
            corpus_dir="static_frame/test/spinwright",
        )
        == expected.resolve()
    )


# ---------------------------------------------------------------------------
# to_stem
# ---------------------------------------------------------------------------


def test_to_stem_strips_dir_and_extension():
    assert to_stem("foo") == "foo"
    assert to_stem("foo.py") == "foo"
    assert to_stem("spinwright/foo.py") == "foo"
    assert to_stem("/abs/spinwright/foo.py") == "foo"
