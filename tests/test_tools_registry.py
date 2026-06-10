from __future__ import annotations

import sys
from pathlib import Path

import pytest

from spinwright.tools import registry


PY = Path(sys.executable)


def _make_tools(tmp_path: Path):
    workspace = tmp_path / "ws"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    return (
        registry.build_extraction_tools(
            workspace_root=workspace,
            repo_dir=repo,
            venv_python=PY,
        ),
        workspace,
        repo,
    )


def test_registry_returns_expected_tool_names(tmp_path: Path):
    tools, _, _ = _make_tools(tmp_path)
    names = [t.name for t in tools]
    assert names == [
        "list_tests",
        "get_test_source",
        "read_source",
        "write_file",
        "edit_file",
        "run_python",
    ]


def test_registry_schemas_are_valid_objects(tmp_path: Path):
    tools, _, _ = _make_tools(tmp_path)
    for tool in tools:
        assert tool.input_schema["type"] == "object"
        assert "properties" in tool.input_schema
        assert tool.description, f"tool {tool.name} has empty description"


def test_handler_write_file_writes_into_workspace(tmp_path: Path):
    tools, workspace, _ = _make_tools(tmp_path)
    write = next(t for t in tools if t.name == "write_file")
    result = write.handler({"path": "extractions/x.py", "content": "print('hi')\n"})
    assert result["created"] is True
    assert (workspace / "extractions" / "x.py").read_text() == "print('hi')\n"


def test_handler_write_file_rejects_escape(tmp_path: Path):
    tools, _, _ = _make_tools(tmp_path)
    write = next(t for t in tools if t.name == "write_file")
    with pytest.raises(Exception):
        write.handler({"path": "../outside.txt", "content": "x"})


def test_handler_run_python_works(tmp_path: Path):
    tools, _, repo = _make_tools(tmp_path)
    runp = next(t for t in tools if t.name == "run_python")
    result = runp.handler({"code": "print(2 + 2)"})
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "4"


def test_handler_read_source_resolves_qualname(tmp_path: Path):
    tools, _, _ = _make_tools(tmp_path)
    reader = next(t for t in tools if t.name == "read_source")
    result = reader.handler({"qualname": "json.loads"})
    assert "def loads" in result["source"]
