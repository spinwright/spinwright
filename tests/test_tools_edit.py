from __future__ import annotations

from pathlib import Path

import pytest

from spinwright.tools import edit


def test_write_file_creates_new_file(tmp_path: Path):
    res = edit.write_file(tmp_path, "subdir/hello.txt", "hi\n")
    assert res.created is True
    assert (tmp_path / "subdir" / "hello.txt").read_text() == "hi\n"
    assert res.bytes_written == 3


def test_write_file_overwrites_existing(tmp_path: Path):
    (tmp_path / "x.txt").write_text("old\n")
    res = edit.write_file(tmp_path, "x.txt", "new\n")
    assert res.created is False
    assert (tmp_path / "x.txt").read_text() == "new\n"


def test_write_file_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(edit.WorkspaceEscapeError):
        edit.write_file(tmp_path, "../escape.txt", "nope")


def test_write_file_rejects_absolute_outside_workspace(tmp_path: Path):
    other = tmp_path.parent / "definitely-outside.txt"
    with pytest.raises(edit.WorkspaceEscapeError):
        edit.write_file(tmp_path, str(other), "nope")


def test_write_file_accepts_absolute_inside_workspace(tmp_path: Path):
    inside = tmp_path / "inside.txt"
    res = edit.write_file(tmp_path, str(inside), "ok\n")
    assert (tmp_path / "inside.txt").read_text() == "ok\n"
    assert res.created is True


def test_edit_file_unique_match_replaces(tmp_path: Path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    res = edit.edit_file(tmp_path, "a.py", "return 1", "return 42")
    assert (tmp_path / "a.py").read_text() == "def foo():\n    return 42\n"
    assert res.bytes_written > 0


def test_edit_file_rejects_zero_matches(tmp_path: Path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    with pytest.raises(edit.EditMatchError, match="not found"):
        edit.edit_file(tmp_path, "a.py", "return 999", "return 42")


def test_edit_file_rejects_multiple_matches(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\nx = 1\nx = 1\n")
    with pytest.raises(edit.EditMatchError, match="3 locations"):
        edit.edit_file(tmp_path, "a.py", "x = 1", "x = 2")


def test_edit_file_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        edit.edit_file(tmp_path, "no_such_file.py", "x", "y")


def test_edit_file_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(edit.WorkspaceEscapeError):
        edit.edit_file(tmp_path, "../outside.py", "x", "y")
