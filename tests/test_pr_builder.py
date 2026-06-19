from __future__ import annotations

from pathlib import Path

from spinwright.llm.dispatch import ConversationResult, TurnRecord
from spinwright.measurement.types import CallgrindResult, VerifyResult, WalltimeResult
from spinwright.optimization.loop import LoopBaseline, LoopResult
from spinwright.optimization.optimize import OptimizationResult
from spinwright.optimization.regression import RegressionResult
from spinwright.pr import builder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _wt(
    median: float,
    best: float | None = None,
    stddev: float = 0.0,
    iters: int = 1000,
    repeats: int = 5,
) -> WalltimeResult:
    return WalltimeResult(
        best_seconds=best if best is not None else median * 0.95,
        median_seconds=median,
        stddev_seconds=stddev,
        iterations_per_repeat=iters,
        repeats=repeats,
    )


def _cg(inst: int, n: int = 1000) -> CallgrindResult:
    return CallgrindResult(
        instructions=inst,
        autoscale_iterations=n,
        total_inst_at_n_plus_one=inst * (n + 1) + 10000,
        baseline_inst_at_one=inst + 10000,
        output_path="/tmp/x.out",
    )


def _conv(final_text: str, tool_calls=None) -> ConversationResult:
    return ConversationResult(
        stop_reason="end_turn",
        turns=[TurnRecord(role="user", content=[])],
        final_text=final_text,
        tool_calls=tool_calls or [],
    )


def _iteration(
    *,
    accepted: bool,
    diff: str = "",
    walltime_delta: float | None = None,
    callgrind_delta: float | None = None,
    gate: str = "walltime_median",
    final_text: str = "did the thing",
    sha: str | None = None,
    tool_calls=None,
) -> OptimizationResult:
    return OptimizationResult(
        accepted=accepted,
        nodeid_or_extraction="/tmp/ext.py",
        baseline_walltime=_wt(1.0),
        candidate_walltime=_wt(1.0 * (1 - (walltime_delta or 0.0))),
        baseline_callgrind=_cg(1_000_000) if callgrind_delta is not None else None,
        candidate_callgrind=_cg(int(1_000_000 * (1 - callgrind_delta)))
        if callgrind_delta is not None
        else None,
        candidate_verify=VerifyResult(passed=True, error=None),
        relative_improvement=callgrind_delta
        if gate == "callgrind_instructions"
        else walltime_delta,
        relative_walltime_improvement=walltime_delta,
        relative_callgrind_improvement=callgrind_delta,
        gate_metric=gate,
        threshold=0.20,
        diff=diff,
        commit_sha=sha,
        rejection_reason=None,
        conversation=_conv(final_text, tool_calls=tool_calls),
        extraction_path=Path("/tmp/ext.py"),
    )


def _loop_result(
    *iterations: OptimizationResult,
    final_wt: WalltimeResult | None = None,
    final_cg: CallgrindResult | None = None,
) -> LoopResult:
    base_wt = _wt(1.0)
    base_cg = (
        _cg(1_000_000) if any(it.baseline_callgrind for it in iterations) else None
    )
    accepted_indices = [i for i, it in enumerate(iterations) if it.accepted]
    return LoopResult(
        success=True,
        extraction_path=Path("/tmp/ext.py"),
        baseline=LoopBaseline(
            walltime=base_wt,
            callgrind=base_cg,
            callgrind_disabled_reason=None,
            verify=VerifyResult(passed=True, error=None),
        ),
        iterations=list(iterations),
        accepted_indices=accepted_indices,
        final_walltime=final_wt or _wt(1.0),
        final_callgrind=final_cg,
        explored=[],
        stop_reason="max_iterations",
    )


def _meta(
    tmp_path: Path, nodeid: str = "tests/test_x.py::test_foo"
) -> builder.ExtractionMetadata:
    return builder.ExtractionMetadata(
        extraction_path=tmp_path / "spinwright" / "ext.py",
        original_nodeid=nodeid,
        source_commit_sha="abc123def4567890",
        corpus_dir="spinwright",
    )


_DIFF_ONE_FILE = """\
diff --git a/static_frame/core/index.py b/static_frame/core/index.py
index 111..222 100644
--- a/static_frame/core/index.py
+++ b/static_frame/core/index.py
@@ -10,3 +10,3 @@ class Index:
-    def slow(self):
-        return [x for x in self.data]
+    def slow(self):
+        return list(self.data)
"""


# ---------------------------------------------------------------------------
# Title tests
# ---------------------------------------------------------------------------


def test_title_no_survivors(tmp_path: Path):
    loop = _loop_result(_iteration(accepted=False, final_text="nothing"))
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="claude-opus-4-7",
        repo_dir=tmp_path,
    )
    assert pr.title == "spinwright / test_foo / no improvements"


def test_review_attempt_published_when_no_survivors(tmp_path: Path):
    attempt = _iteration(
        accepted=False,
        diff=_DIFF_ONE_FILE,
        walltime_delta=0.05,
        final_text="Replaced the comprehension with list().",
    )
    attempt.rejection_reason = (
        "median walltime improvement 5.00% is below threshold 20%"
    )
    loop = _loop_result(attempt)
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="anthropic/claude-opus-4-7",
        repo_dir=tmp_path,
        review_attempts=(attempt,),
    )
    assert pr.accepted_count == 0
    # Review-mode title carries the attempted delta as a signed percentage.
    assert pr.title.startswith("spinwright / test_foo / review")
    assert "-5%" in pr.title
    assert "tested-but-unaccepted" in pr.body
    assert "Replaced the comprehension" in pr.body  # the model's reasoning
    assert "below threshold" in pr.body  # why it wasn't accepted
    assert "```diff" in pr.body  # the change itself


def test_review_attempt_ignored_when_a_survivor_exists(tmp_path: Path):
    survivor = _iteration(
        accepted=True, diff=_DIFF_ONE_FILE, walltime_delta=0.30, sha="deadbeef"
    )
    rejected = _iteration(accepted=False, diff=_DIFF_ONE_FILE, walltime_delta=0.05)
    loop = _loop_result(survivor, rejected)
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="m",
        repo_dir=tmp_path,
        review_attempts=(rejected,),
    )
    # A real survivor takes precedence; the review path is not taken.
    assert pr.accepted_count == 1
    assert "tested-but-unaccepted" not in pr.body


def test_title_single_survivor_uses_fixed_three_segment_form(tmp_path: Path):
    """No more LLM-authored description in the title — older versions
    sometimes produced titles like 'All 1861 tests pass' from the model's
    final assistant text. Fixed form ``spinwright / <test> / <±delta>``."""
    loop = _loop_result(
        _iteration(
            accepted=True,
            diff=_DIFF_ONE_FILE,
            walltime_delta=0.30,
            gate="walltime_median",
            final_text="Swapped the list comprehension for a builtin list() call.",
            sha="aaa111",
        ),
        final_wt=_wt(0.70),
    )
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="claude-opus-4-7",
        repo_dir=tmp_path,
    )
    assert pr.title == "spinwright / test_foo / -30%"


def test_title_multi_survivor_uses_total_delta(tmp_path: Path):
    """Multiple survivors collapse to the same fixed format — the body has the
    per-patch breakdown, the title just carries the cumulative number."""
    diff1 = _DIFF_ONE_FILE
    diff2 = _DIFF_ONE_FILE.replace("index.py", "frame.py")
    loop = _loop_result(
        _iteration(accepted=True, diff=diff1, walltime_delta=0.20, sha="aaa"),
        _iteration(accepted=True, diff=diff2, walltime_delta=0.30, sha="bbb"),
        final_wt=_wt(0.50),  # 1.0 → 0.5 = 50% total wallclock reduction
    )
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="claude-opus-4-7",
        repo_dir=tmp_path,
    )
    assert pr.title == "spinwright / test_foo / -50%"


def test_title_callgrind_delta_unitless(tmp_path: Path):
    """The title no longer carries the metric name (instructions vs wallclock)
    — the body's measurements table makes that distinction explicitly. The
    title just reports the signed percentage for whichever gate was used."""
    loop = _loop_result(
        _iteration(
            accepted=True,
            diff=_DIFF_ONE_FILE,
            walltime_delta=0.10,
            callgrind_delta=0.40,
            gate="callgrind_instructions",
            final_text="A clear win.",
            sha="aaa",
        ),
        final_wt=_wt(0.90),
        final_cg=_cg(600_000),
    )
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="claude-opus-4-7",
        repo_dir=tmp_path,
    )
    assert pr.title == "spinwright / test_foo / -40%"


# ---------------------------------------------------------------------------
# Body tests
# ---------------------------------------------------------------------------


def test_body_has_all_sections(tmp_path: Path):
    loop = _loop_result(
        _iteration(
            accepted=True,
            diff=_DIFF_ONE_FILE,
            walltime_delta=0.25,
            sha="aaa111bbb",
            final_text="Removed an unnecessary copy.",
        ),
        final_wt=_wt(0.75),
    )
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="run_20260529_120000",
        model="claude-opus-4-7",
        repo_dir=tmp_path,
    )
    body = pr.body
    assert "## Summary" in body
    assert "## Measurements" in body
    assert "## Test" in body
    assert "## Bottlenecks and Changes" in body
    assert "## Notes" in body
    # Test section mentions nodeid + source SHA
    assert "tests/test_x.py::test_foo" in body
    assert "abc123def" in body
    # Bottleneck section has summary + diff
    assert "Removed an unnecessary copy" in body
    assert "```diff" in body
    assert "list(self.data)" in body
    # Notes section has version + run id
    assert "Run ID: `run_20260529_120000`" in body


def test_body_quotes_dropped_patches(tmp_path: Path):
    loop = _loop_result(
        _iteration(
            accepted=True,
            diff=_DIFF_ONE_FILE,
            walltime_delta=0.30,
            sha="aaa",
            final_text="A",
        ),
        _iteration(
            accepted=True,
            diff=_DIFF_ONE_FILE.replace("index.py", "frame.py"),
            walltime_delta=0.25,
            sha="bbb",
            final_text="B",
        ),
        final_wt=_wt(0.50),
    )
    reg = RegressionResult(
        passed=True,
        dropped_commits=["bbb"],
        final_pytest_output="all good",
        fallback_used="linear_revert",
    )
    pr = builder.build_pr(
        loop_result=loop,
        regression=reg,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="claude-opus-4-7",
        repo_dir=tmp_path,
    )
    assert pr.accepted_count == 1
    assert pr.dropped_count == 1
    assert "## Dropped patches" in pr.body
    assert "bbb" in pr.body
    assert "linear_revert" in pr.body


def test_body_no_survivors_section_is_compact(tmp_path: Path):
    loop = _loop_result(_iteration(accepted=False, final_text="nope"))
    pr = builder.build_pr(
        loop_result=loop,
        regression=None,
        extraction=_meta(tmp_path),
        run_id="r1",
        model="claude-opus-4-7",
        repo_dir=tmp_path,
    )
    assert "no improvements clearing the gate threshold" in pr.body
    assert "informational only" in pr.body
    assert pr.accepted_count == 0


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_module_from_path_with_init():
    assert builder._module_from_path("a/pkg/__init__.py") == "pkg"


def test_module_from_path_strips_diff_prefix():
    assert (
        builder._module_from_path("a/static_frame/core/index.py")
        == "static_frame.core.index"
    )


def test_diff_paths_extracts_files():
    paths = builder.diff_rel_paths(_DIFF_ONE_FILE)
    assert paths == ["static_frame/core/index.py"]


def test_format_delta_is_signed_bare_percent():
    """A 30% reduction renders as `-30%`; a regression renders as `+12%`.
    No metric label in the string — titles append it explicitly."""
    assert builder._format_delta(0.30) == "-30%"
    assert builder._format_delta(-0.12) == "+12%"
    assert builder._format_delta(0.0) == "+0%"
