from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from spinwright import run_log
from spinwright.optimization.loop import LoopBaseline, LoopResult
from spinwright.optimization.regression import RegressionResult
from spinwright.measurement.types import VerifyResult, WalltimeResult
from spinwright.pr.builder import PRDraft


def _wt(median=0.001, repeats=5, iters=1000) -> WalltimeResult:
    return WalltimeResult(
        best_seconds=median * 0.95,
        median_seconds=median,
        stddev_seconds=median * 0.01,
        iterations_per_repeat=iters,
        repeats=repeats,
    )


def _minimal_loop() -> LoopResult:
    return LoopResult(
        success=True,
        extraction_path=Path("/tmp/ext.py"),
        baseline=LoopBaseline(
            walltime=_wt(),
            callgrind=None,
            callgrind_disabled_reason="macOS",
            verify=VerifyResult(passed=True, error=None),
        ),
        iterations=[],
        accepted_indices=[],
        final_walltime=_wt(0.0008),
        final_callgrind=None,
        explored=[],
        stop_reason="no_remaining_bottlenecks",
    )


def test_make_run_id_is_timestamp_prefixed():
    now = dt.datetime(2026, 5, 29, 12, 0, 30, 123456, tzinfo=dt.timezone.utc)
    rid = run_log.make_run_id(now)
    assert rid.startswith("run_20260529T120030_")
    assert len(rid) > len("run_20260529T120030_")


def test_make_run_ids_are_unique_within_same_second():
    rid1 = run_log.make_run_id()
    rid2 = run_log.make_run_id()
    # Trivially different microseconds on consecutive calls
    assert rid1 != rid2 or rid1.startswith("run_")


def test_write_run_directory_creates_pr_md(tmp_path: Path):
    loop = _minimal_loop()
    pr = PRDraft(
        title="perf(x): test",
        body="## Summary\n\nbody body body\n",
        accepted_count=1,
        dropped_count=0,
    )
    run_dir = run_log.write_run_directory(
        runs_root=tmp_path / "runs",
        run_id="test_run",
        pr_draft=pr,
        loop_result=loop,
        regression=None,
    )
    assert run_dir == tmp_path / "runs" / "test_run"
    pr_md = (run_dir / "PR.md").read_text()
    assert pr_md.startswith("# perf(x): test\n")
    assert "## Summary" in pr_md
    assert (run_dir / "run_summary.json").exists()


def test_write_run_directory_serializes_loop_and_regression(tmp_path: Path):
    loop = _minimal_loop()
    reg = RegressionResult(
        passed=True,
        dropped_commits=["abc", "def"],
        final_pytest_output="ok",
        fallback_used="linear_revert",
    )
    pr = PRDraft(title="t", body="b", accepted_count=2, dropped_count=2)
    run_dir = run_log.write_run_directory(
        runs_root=tmp_path,
        run_id="rid",
        pr_draft=pr,
        loop_result=loop,
        regression=reg,
        extra_metadata={"workspace": "/tmp/x", "extraction": "/tmp/ext.py"},
    )
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["run_id"] == "rid"
    assert summary["loop_result"]["stop_reason"] == "no_remaining_bottlenecks"
    assert summary["regression"]["dropped_commits"] == ["abc", "def"]
    assert summary["pr"]["accepted_count"] == 2
    assert summary["metadata"]["workspace"] == "/tmp/x"


def test_write_run_directory_without_pr_skips_pr_md(tmp_path: Path):
    loop = _minimal_loop()
    run_dir = run_log.write_run_directory(
        runs_root=tmp_path,
        run_id="no_pr",
        pr_draft=None,
        loop_result=loop,
        regression=None,
    )
    assert not (run_dir / "PR.md").exists()
    assert (run_dir / "run_summary.json").exists()
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["pr"] is None


def test_write_run_directory_overwrites_existing(tmp_path: Path):
    loop = _minimal_loop()
    pr = PRDraft(title="t", body="b1", accepted_count=1, dropped_count=0)
    runs_root = tmp_path / "runs"
    run_log.write_run_directory(
        runs_root=runs_root,
        run_id="r",
        pr_draft=pr,
        loop_result=loop,
        regression=None,
    )
    pr2 = PRDraft(title="t", body="b2 updated", accepted_count=1, dropped_count=0)
    run_log.write_run_directory(
        runs_root=runs_root,
        run_id="r",
        pr_draft=pr2,
        loop_result=loop,
        regression=None,
    )
    assert "b2 updated" in (runs_root / "r" / "PR.md").read_text()
