from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from spinwright.tools import source


PY = Path(sys.executable)


# Source-level tests (get_test_source) use a tmp Python file; no venv needed.


def _write_tests_file(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "test_mod.py"
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return p


def test_get_test_source_returns_function_slice(tmp_path: Path):
    _write_tests_file(
        tmp_path,
        """
        import pytest

        @pytest.mark.smoke
        def test_alpha():
            assert 1 + 1 == 2

        def test_beta():
            assert True
    """,
    )
    result = source.get_test_source(tmp_path, "test_mod.py::test_alpha")
    assert result["kind"] == "function"
    assert result["class_name"] is None
    assert (
        result["lineno"] == 4
    )  # ast.FunctionDef.lineno is the `def` line, not the decorator
    assert "def test_alpha" in result["source"]
    assert "test_beta" not in result["source"]
    assert any("@pytest.mark.smoke" in d for d in result["decorators"])


def test_get_test_source_returns_method_slice(tmp_path: Path):
    _write_tests_file(
        tmp_path,
        """
        import unittest

        class TestThing(unittest.TestCase):
            def test_method(self):
                self.assertEqual(2 + 2, 4)
    """,
    )
    result = source.get_test_source(tmp_path, "test_mod.py::TestThing::test_method")
    assert result["kind"] == "method"
    assert result["class_name"] == "TestThing"
    assert "def test_method" in result["source"]


def test_get_test_source_strips_parametrize_suffix(tmp_path: Path):
    _write_tests_file(
        tmp_path,
        """
        import pytest

        @pytest.mark.parametrize('n', [1, 2])
        def test_p(n):
            assert n > 0
    """,
    )
    result = source.get_test_source(tmp_path, "test_mod.py::test_p[1]")
    assert "def test_p" in result["source"]


def test_get_test_source_raises_for_missing_test(tmp_path: Path):
    _write_tests_file(tmp_path, "def test_x(): pass\n")
    with pytest.raises(LookupError):
        source.get_test_source(tmp_path, "test_mod.py::test_missing")


def test_get_test_source_raises_for_missing_nodeid_component(tmp_path: Path):
    _ = _write_tests_file(tmp_path, "def test_x(): pass\n")
    with pytest.raises(ValueError):
        source.get_test_source(tmp_path, "test_mod.py")


# ---------------------------------------------------------------------------
# read_source uses a subprocess of the current Python; resolves stdlib symbols
# without needing a target-repo venv.
# ---------------------------------------------------------------------------


def test_read_source_resolves_stdlib_function():
    res = source.read_source(PY, "json.loads")
    assert res["module"] == "json"
    assert "def loads" in res["source"]
    assert res["path"].endswith("json/__init__.py")


def test_read_source_resolves_class_method():
    res = source.read_source(PY, "pathlib.Path.is_absolute")
    assert "def is_absolute" in res["source"]


def test_read_source_raises_for_unknown_qualname():
    with pytest.raises(LookupError):
        source.read_source(PY, "no.such.module.attr")


# ---------------------------------------------------------------------------
# list_tests goes through pytest in a venv. We build a minimal venv-less
# scenario by using the test runner's interpreter and a tiny ad-hoc tests dir.
# ---------------------------------------------------------------------------


def _init_tiny_pytest_repo(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "def test_one(): assert True\ndef test_two(): assert True\n"
    )
    (tmp_path / "tests" / "test_b.py").write_text("def test_three(): assert True\n")
    return tmp_path


def test_list_tests_enumerates_nodeids(tmp_path: Path):
    repo = _init_tiny_pytest_repo(tmp_path)
    nodeids = source.list_tests(PY, repo)
    assert "tests/test_a.py::test_one" in nodeids
    assert "tests/test_a.py::test_two" in nodeids
    assert "tests/test_b.py::test_three" in nodeids
    assert len(nodeids) == 3


def test_list_tests_supports_pattern(tmp_path: Path):
    repo = _init_tiny_pytest_repo(tmp_path)
    nodeids = source.list_tests(PY, repo, pattern="test_three")
    assert nodeids == ["tests/test_b.py::test_three"]
