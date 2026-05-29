from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from spinwright.measurement.runner import DriverError
from spinwright.profiling import cprofile


PY = Path(sys.executable)


def _write_extraction(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "extract.py"
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return p


_EXTRACTION_TEMPLATE = """
def setup():
    return {"n": 100}

def busy(state):
    total = 0
    for i in range(state["n"]):
        total += i * i
    state["_total"] = total

def run(state):
    busy(state)

def verify(state):
    assert state["_total"] == sum(i * i for i in range(state["n"]))
"""


def test_profile_returns_entries_and_verify_passes(tmp_path: Path):
    ext = _write_extraction(tmp_path, _EXTRACTION_TEMPLATE)
    result = cprofile.profile_cprofile(PY, ext, iterations=200)
    assert result.verify_passed
    assert result.iterations == 200
    assert result.total_seconds > 0
    assert len(result.entries) > 0
    funcnames = {e.funcname for e in result.entries}
    assert "busy" in funcnames
    assert "run" in funcnames


def test_top_entries_sorts_by_cumtime(tmp_path: Path):
    ext = _write_extraction(tmp_path, _EXTRACTION_TEMPLATE)
    result = cprofile.profile_cprofile(PY, ext, iterations=200)
    top = cprofile.top_entries(result, by="cumtime", limit=5)
    assert len(top) <= 5
    # Sorted descending
    times = [e.cumtime for e in top]
    assert times == sorted(times, reverse=True)


def test_top_entries_rejects_unknown_sort_key(tmp_path: Path):
    ext = _write_extraction(tmp_path, _EXTRACTION_TEMPLATE)
    result = cprofile.profile_cprofile(PY, ext, iterations=10)
    with pytest.raises(ValueError, match="unknown sort key"):
        cprofile.top_entries(result, by="bogus", limit=5)


def test_exclude_paths_drops_matching_entries(tmp_path: Path):
    ext = _write_extraction(tmp_path, _EXTRACTION_TEMPLATE)
    unfiltered = cprofile.profile_cprofile(PY, ext, iterations=200)
    # The extraction module is under tmp_path; excluding it should drop the
    # `run`/`busy` entries (which are the only user-code entries) while
    # leaving anything outside that path (built-in/runtime helpers, if any).
    filtered = cprofile.profile_cprofile(
        PY, ext, iterations=200, exclude_paths=(str(tmp_path),)
    )
    filtered_funcs = {e.funcname for e in filtered.entries}
    assert "busy" not in filtered_funcs
    assert "run" not in filtered_funcs
    # Inverse: an exclude that doesn't match anything keeps the user-code entries.
    unmatched = cprofile.profile_cprofile(
        PY, ext, iterations=200, exclude_paths=("definitely-not-a-real-path",)
    )
    assert {"busy", "run"} <= {e.funcname for e in unmatched.entries}
    assert len(unmatched.entries) == len(unfiltered.entries)


def test_verify_failure_is_reported(tmp_path: Path):
    ext = _write_extraction(tmp_path, """
        def setup(): return {}
        def run(state): pass
        def verify(state): raise AssertionError('nope')
    """)
    result = cprofile.profile_cprofile(PY, ext, iterations=10)
    assert result.verify_passed is False
    assert "AssertionError" in (result.verify_error or "")
    # But the profile entries still came back
    assert len(result.entries) > 0


def test_driver_error_propagates(tmp_path: Path):
    ext = _write_extraction(tmp_path, "def setup(): raise RuntimeError('boom')\n"
                                       "def run(s): pass\n"
                                       "def verify(s): pass\n")
    with pytest.raises(DriverError):
        cprofile.profile_cprofile(PY, ext, iterations=10)
