from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from spinwright.extraction import eligibility


def _write(tmp_path: Path, source: str, name: str = "test_mod.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(source).lstrip("\n"))
    return p


def _codes(result: eligibility.EligibilityResult) -> set[str]:
    return {r.code for r in result.reasons}


# ---------------------------------------------------------------------------
# Eligible cases
# ---------------------------------------------------------------------------


def test_plain_pytest_function_is_eligible(tmp_path: Path):
    src = _write(tmp_path, """
        def test_simple():
            assert 1 + 1 == 2
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_simple")
    assert r.eligible, r.reasons


def test_tmp_path_fixture_is_allowed(tmp_path: Path):
    src = _write(tmp_path, """
        def test_with_tmp(tmp_path):
            (tmp_path / 'x').mkdir()
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_with_tmp")
    assert r.eligible, r.reasons


def test_unittest_method_with_trivial_setup_is_eligible(tmp_path: Path):
    src = _write(tmp_path, """
        import unittest

        class TestThing(unittest.TestCase):
            def setUp(self):
                self.x = 5
                self.y = (1, 2, 3)

            def test_it(self):
                self.assertEqual(self.x, 5)
    """)
    r = eligibility.check(src, "tests/test_mod.py::TestThing::test_it")
    assert r.eligible, r.reasons


def test_seeded_random_is_eligible(tmp_path: Path):
    src = _write(tmp_path, """
        import random

        def test_seeded():
            random.seed(42)
            assert 0 <= random.random() < 1
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_seeded")
    assert r.eligible, r.reasons


def test_seeded_numpy_random_is_eligible(tmp_path: Path):
    src = _write(tmp_path, """
        import numpy as np

        def test_np_seeded():
            np.random.seed(0)
            _ = np.random.rand(10)
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_np_seeded")
    assert r.eligible, r.reasons


# ---------------------------------------------------------------------------
# Ineligible: fixtures
# ---------------------------------------------------------------------------


def test_pytest_fixture_arg_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        def test_with_fixture(my_db):
            assert my_db is not None
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_with_fixture")
    assert not r.eligible
    assert "fixture_arg" in _codes(r)


# ---------------------------------------------------------------------------
# Ineligible: parametrize / skip markers
# ---------------------------------------------------------------------------


def test_parametrize_marker_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        import pytest

        @pytest.mark.parametrize('n', [1, 2, 3])
        def test_p(n):
            assert n > 0
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_p")
    assert not r.eligible
    assert "pytest_marker" in _codes(r)


def test_parametrize_nodeid_with_param_suffix_still_resolves(tmp_path: Path):
    # pytest emits one nodeid per parameter ("test_p[1]", "test_p[2]"). The
    # checker must strip "[...]" so it finds the function and emits
    # "pytest_marker" rather than "not_found".
    src = _write(tmp_path, """
        import pytest

        @pytest.mark.parametrize('n', [1, 2, 3])
        def test_p(n):
            assert n > 0
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_p[1]")
    assert not r.eligible
    codes = _codes(r)
    assert "pytest_marker" in codes
    assert "not_found" not in codes


def test_skip_marker_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        import pytest

        @pytest.mark.skip(reason='broken')
        def test_skipped():
            pass
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_skipped")
    assert not r.eligible
    assert "pytest_marker" in _codes(r)


# ---------------------------------------------------------------------------
# Ineligible: hypothesis
# ---------------------------------------------------------------------------


def test_given_decorator_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        from hypothesis import given, strategies as st

        @given(st.integers())
        def test_h(n):
            assert isinstance(n, int)
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_h")
    assert not r.eligible
    assert "hypothesis_decorator" in _codes(r) or "hypothesis_import" in _codes(r)


# ---------------------------------------------------------------------------
# Ineligible: network / subprocess / filesystem
# ---------------------------------------------------------------------------


def test_subprocess_call_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        import subprocess

        def test_sp():
            subprocess.run(['ls'], check=True)
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_sp")
    assert not r.eligible
    assert "subprocess_call" in _codes(r)


def test_network_call_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        import requests

        def test_net():
            requests.get('http://example.com').text
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_net")
    assert not r.eligible
    assert "network_call" in _codes(r)


def test_subprocess_imported_but_not_used_is_eligible(tmp_path: Path):
    # Realistic case: a test module imports subprocess for one test, but the
    # specific test we target doesn't use it. Module-level import alone must
    # not reject — SPEC §5.2 ineligibility is on call-site usage.
    src = _write(tmp_path, """
        import subprocess

        def test_unrelated():
            assert 2 + 2 == 4

        def test_uses_sp():
            subprocess.run(['true'])
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_unrelated")
    assert r.eligible, r.reasons


def test_open_call_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        def test_open():
            with open('/etc/hosts') as f:
                f.read()
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_open")
    assert not r.eligible
    assert "filesystem_open" in _codes(r)


# ---------------------------------------------------------------------------
# Ineligible: unseeded randomness
# ---------------------------------------------------------------------------


def test_unseeded_random_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        import random

        def test_unseeded():
            _ = random.random()
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_unseeded")
    assert not r.eligible
    assert "unseeded_random" in _codes(r)


def test_unseeded_np_random_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        import numpy as np

        def test_np_unseeded():
            _ = np.random.rand(10)
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_np_unseeded")
    assert not r.eligible
    assert "unseeded_np_random" in _codes(r)


# ---------------------------------------------------------------------------
# Ineligible: conftest imports
# ---------------------------------------------------------------------------


def test_conftest_import_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        from .conftest import helper

        def test_x():
            helper()
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_x")
    assert not r.eligible
    assert "conftest_import" in _codes(r)


def test_conftest_import_allowed_when_configured(tmp_path: Path):
    src = _write(tmp_path, """
        from .conftest import helper

        def test_x():
            helper()
    """)
    r = eligibility.check(
        src,
        "tests/test_mod.py::test_x",
        allow_pure_conftest_imports=True,
    )
    # conftest_import is suppressed, but helper() can't be resolved so other
    # reasons (if any) might fire — assert specifically that conftest_import is gone.
    assert "conftest_import" not in _codes(r)


# ---------------------------------------------------------------------------
# Ineligible: non-trivial unittest lifecycle
# ---------------------------------------------------------------------------


def test_nontrivial_setup_rejected(tmp_path: Path):
    src = _write(tmp_path, """
        import unittest

        class TestThing(unittest.TestCase):
            def setUp(self):
                for i in range(10):
                    self.records.append(i)

            def test_x(self):
                self.assertTrue(self.records)
    """)
    r = eligibility.check(src, "tests/test_mod.py::TestThing::test_x")
    assert not r.eligible
    assert "unittest_lifecycle_nontrivial" in _codes(r)


def test_super_lifecycle_call_is_trivial(tmp_path: Path):
    src = _write(tmp_path, """
        import unittest

        class TestThing(unittest.TestCase):
            def setUp(self):
                super().setUp()
                self.x = 5

            def test_x(self):
                self.assertEqual(self.x, 5)
    """)
    r = eligibility.check(src, "tests/test_mod.py::TestThing::test_x")
    assert r.eligible, r.reasons


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_missing_test_reported(tmp_path: Path):
    src = _write(tmp_path, "def test_other(): pass\n")
    r = eligibility.check(src, "tests/test_mod.py::test_nope")
    assert not r.eligible
    assert "not_found" in _codes(r)


def test_nodeid_without_test_component_rejected(tmp_path: Path):
    src = _write(tmp_path, "def test_x(): pass\n")
    r = eligibility.check(src, "tests/test_mod.py")
    assert not r.eligible
    assert "nodeid_invalid" in _codes(r)


def test_reasons_include_lineno(tmp_path: Path):
    src = _write(tmp_path, """
        import subprocess

        def test_sp():
            subprocess.run(['ls'])
    """)
    r = eligibility.check(src, "tests/test_mod.py::test_sp")
    # Every reason that fired came from a specific line — lineno should be set.
    assert all(reason.lineno is not None for reason in r.reasons), r.reasons
