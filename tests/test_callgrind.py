from __future__ import annotations

import sys
from pathlib import Path

import pytest

from spinwright.measurement import callgrind
from spinwright.measurement.auto_scale import autoscale_iterations


# ---------------------------------------------------------------------------
# parse_summary
# ---------------------------------------------------------------------------


_SAMPLE_OUTPUT = """\
version: 1
creator: callgrind-3.22.0
pid: 12345
cmd: python driver.py
desc: I1 cache:
desc: D1 cache:
positions: line
events: Ir

fl=driver.py
fn=run
1 100
2 200

totals: 300
summary: 300
"""


def test_parse_summary_reads_instruction_total(tmp_path: Path):
    p = tmp_path / "out.cg"
    p.write_text(_SAMPLE_OUTPUT)
    assert callgrind.parse_summary(p) == 300


def test_parse_summary_returns_last_when_multiple(tmp_path: Path):
    # Some callgrind outputs (multiple snapshots) emit several `summary:` lines.
    # The orchestrator wants the latest, which represents the full run.
    p = tmp_path / "out.cg"
    p.write_text(_SAMPLE_OUTPUT + "\nsummary: 999\n")
    assert callgrind.parse_summary(p) == 999


def test_parse_summary_raises_on_malformed(tmp_path: Path):
    p = tmp_path / "out.cg"
    p.write_text("version: 1\nno summary here\n")
    with pytest.raises(ValueError, match="no 'summary:'"):
        callgrind.parse_summary(p)


# ---------------------------------------------------------------------------
# autoscale
# ---------------------------------------------------------------------------


def test_autoscale_picks_one_for_zero_time():
    assert autoscale_iterations(0.0, 1_000_000_000) == 1


def test_autoscale_grows_with_min_instructions():
    # 1us per call ≈ 1000 estimated instructions; want 1e9 total → 1e6 iters.
    n = autoscale_iterations(1e-6, 1_000_000_000)
    assert 900_000 <= n <= 1_100_000


def test_autoscale_falls_for_slow_calls():
    # 100ms per call ≈ 1e8 estimated instructions; want 1e9 total → ~10 iters.
    n = autoscale_iterations(0.1, 1_000_000_000)
    assert 8 <= n <= 12


def test_autoscale_caps_unbounded_estimates():
    # A microscopic probe time can blow up the estimate. The cap keeps us sane.
    n = autoscale_iterations(1e-15, 1_000_000_000, cap=10_000)
    assert n == 10_000


# ---------------------------------------------------------------------------
# Platform gating
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="checks the macOS-side guard only")
def test_measure_callgrind_refuses_on_non_linux(tmp_path: Path):
    ext = tmp_path / "ext.py"
    ext.write_text("def setup(): return {}\ndef run(s): pass\ndef verify(s): pass\n")
    with pytest.raises(callgrind.CallgrindUnavailable, match="Linux"):
        callgrind.measure_callgrind(Path(sys.executable), ext)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux + valgrind")
def test_measure_callgrind_refuses_when_valgrind_missing(tmp_path: Path):
    ext = tmp_path / "ext.py"
    ext.write_text("def setup(): return {}\ndef run(s): pass\ndef verify(s): pass\n")
    with pytest.raises(callgrind.CallgrindUnavailable, match="valgrind binary"):
        callgrind.measure_callgrind(
            Path(sys.executable), ext,
            valgrind_path="/nonexistent/valgrind",
        )
