from __future__ import annotations

from spinwright.extraction.discovery import NodeDuration, _parse_durations


SAMPLE_OUTPUT = """\
collected 4 items

tests/test_a.py ...                                                     [ 75%]
tests/test_b.py F                                                       [100%]

============================= slowest 10 durations =============================
0.50s call     tests/test_a.py::test_slow
0.10s call     tests/test_b.py::test_failing
0.05s call     tests/test_a.py::test_fast
0.04s setup    tests/test_a.py::test_slow
0.01s teardown tests/test_a.py::test_slow
==================== 3 passed, 1 failed in 0.62s ====================
"""


def test_parse_durations_extracts_all_phases():
    durations = _parse_durations(SAMPLE_OUTPUT)
    phases = [(d.phase, d.nodeid, d.seconds) for d in durations]
    assert ("call", "tests/test_a.py::test_slow", 0.50) in phases
    assert ("call", "tests/test_b.py::test_failing", 0.10) in phases
    assert ("call", "tests/test_a.py::test_fast", 0.05) in phases
    assert ("setup", "tests/test_a.py::test_slow", 0.04) in phases
    assert ("teardown", "tests/test_a.py::test_slow", 0.01) in phases


def test_parse_durations_ignores_lines_outside_section():
    # No "durations" header → nothing parsed.
    assert _parse_durations("0.50s call tests/test_a.py::test_slow\n") == []


def test_parse_durations_stops_at_trailing_separator():
    # The trailing "===... passed" line must not be parsed as a duration.
    durations = _parse_durations(SAMPLE_OUTPUT)
    assert all("passed" not in d.nodeid for d in durations)


def test_parse_durations_handles_unittest_style_nodeids():
    output = """\
============================= slowest 10 durations =============================
1.25s call     tests/test_x.py::TestThing::test_method
"""
    durations = _parse_durations(output)
    assert durations == [
        NodeDuration(
            nodeid="tests/test_x.py::TestThing::test_method",
            seconds=1.25,
            phase="call",
        )
    ]


def test_parse_durations_handles_no_count_header():
    # Modern pytest with --durations=0 emits "slowest durations" (no number).
    output = """\
============================== slowest durations ===============================
0.51s call     tests/test_d.py::test_slow
0.20s call     tests/test_d.py::test_medium

(4 durations < 0.005s hidden.  Use -vv to show these durations.)
2 passed in 0.72s
"""
    durations = _parse_durations(output)
    nodeids = [d.nodeid for d in durations if d.phase == "call"]
    assert "tests/test_d.py::test_slow" in nodeids
    assert "tests/test_d.py::test_medium" in nodeids
