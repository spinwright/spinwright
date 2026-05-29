from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from spinwright.measurement import walltime
from spinwright.measurement.runner import DriverError


# All tests use the test runner's interpreter as the "target venv" Python — the
# extraction modules are pure stdlib so any 3.11+ interpreter works.
PY = Path(sys.executable)


def _write_extraction(tmp_path: Path, body: str, name: str = "extract.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return p


def test_measure_returns_sensible_walltime(tmp_path: Path):
    ext = _write_extraction(tmp_path, """
        def setup():
            return {'n': 1000}

        def run(state):
            total = 0
            for i in range(state['n']):
                total += i
            state['_last_total'] = total

        def verify(state):
            assert state['_last_total'] == sum(range(state['n']))
    """)
    wt, vr = walltime.measure(PY, ext, repeats=3)
    assert vr.passed
    assert vr.error is None
    assert wt.repeats == 3
    assert wt.iterations_per_repeat >= 1
    assert wt.best_seconds > 0
    assert wt.median_seconds >= wt.best_seconds
    assert wt.stddev_seconds >= 0


def test_measure_reports_verify_failure_without_raising(tmp_path: Path):
    ext = _write_extraction(tmp_path, """
        def setup():
            return {'x': 1}

        def run(state):
            state['x'] += 1

        def verify(state):
            assert state['x'] == -999, f"got {state['x']}"
    """)
    wt, vr = walltime.measure(PY, ext, repeats=2)
    assert vr.passed is False
    assert vr.error is not None
    assert "AssertionError" in vr.error
    # We still got walltime numbers even though verify failed.
    assert wt.repeats == 2


def test_measure_raises_drivererror_when_setup_explodes(tmp_path: Path):
    ext = _write_extraction(tmp_path, """
        def setup():
            raise RuntimeError("boom in setup")

        def run(state):
            pass

        def verify(state):
            pass
    """)
    with pytest.raises(DriverError) as ei:
        walltime.measure(PY, ext, repeats=2)
    assert ei.value.returncode != 0
    assert "boom in setup" in ei.value.stderr


def test_measure_raises_drivererror_when_extraction_unimportable(tmp_path: Path):
    ext = _write_extraction(tmp_path, "def setup(): pass\n  bad-indent\n")
    with pytest.raises(DriverError):
        walltime.measure(PY, ext, repeats=2)


def test_repeats_round_trips(tmp_path: Path):
    ext = _write_extraction(tmp_path, """
        def setup(): return {}
        def run(state): pass
        def verify(state): pass
    """)
    wt, vr = walltime.measure(PY, ext, repeats=7)
    assert wt.repeats == 7
    assert vr.passed
