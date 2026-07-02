from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import spinwright
from spinwright.optimization.loop import LoopResult
from spinwright.optimization.optimize import OptimizationResult
from spinwright.optimization.regression import RegressionResult

if TYPE_CHECKING:
    from spinwright.measurement.types import CallgrindResult, WalltimeResult


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionMetadata:
    """Identifies the extraction the optimizations were applied against."""

    extraction_path: Path
    original_nodeid: str | None
    source_commit_sha: str
    corpus_dir: str


@dataclass(frozen=True)
class PRDraft:
    title: str
    body: str
    accepted_count: int  # patches still in after regression
    dropped_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pr(
    *,
    loop_result: LoopResult,
    regression: RegressionResult | None,
    extraction: ExtractionMetadata,
    run_id: str,
    model: str,
    repo_dir: Path,
    review_attempts: tuple[OptimizationResult, ...] = (),
) -> PRDraft:
    """Render a PR title + body from an optimization run's outcome.

    Patches dropped by the regression check are filtered out; only the
    survivors are quoted in the body. If no patches survive, the title is
    still computed but the body reads as a "nothing to ship" report so the
    caller can decide whether to skip the PR entirely.

    ``review_attempts`` are unaccepted-but-committed attempts published under
    ``--always-publish`` (see ``publish.commit_attempt_for_review``). They only
    matter when there are no survivors: the body then renders their diff,
    summary, measured impact, and rejection reason so the model's tested change
    and reasoning can be reviewed even though it didn't clear the gate.
    """
    dropped_set = set(regression.dropped_commits) if regression else set()
    survivors: list[OptimizationResult] = []
    for idx in loop_result.accepted_indices:
        it = loop_result.iterations[idx]
        if it.commit_sha and it.commit_sha not in dropped_set:
            survivors.append(it)

    # Review attempts are only surfaced when nothing cleared the gate.
    review = () if survivors else tuple(review_attempts)

    # How deltas are sourced depends on the outcome:
    #   - Survivors: baseline → FINAL on-disk state (the loop's final readings).
    #     We can't trivially re-measure here, so those are the best-available
    #     figure. If a patch was dropped during regression the displayed deltas
    #     overstate improvement — we mark this in the body.
    #   - No survivors, review attempt: the attempt was reverted, so the final
    #     state equals baseline. Title and table instead report the *attempt's*
    #     own before/after (what it measured with the patch applied); the body
    #     labels this as a reverted attempt.
    title = _build_title(loop_result, survivors, extraction, review, model=model)
    body = _build_body(
        loop_result=loop_result,
        survivors=survivors,
        regression=regression,
        extraction=extraction,
        run_id=run_id,
        model=model,
        repo_dir=repo_dir,
        review_attempts=review,
    )
    return PRDraft(
        title=title,
        body=body,
        accepted_count=len(survivors),
        dropped_count=len(dropped_set),
    )


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------


def _build_title(
    loop_result: LoopResult,
    survivors: list[OptimizationResult],
    extraction: ExtractionMetadata,
    review_attempts: tuple[OptimizationResult, ...] = (),
    *,
    model: str,
) -> str:
    """Format: ``spinwright / <test_name> / <±delta> / <model>``.

    Older versions of this function tried to inject an LLM-authored
    description into the title (first sentence of the final assistant turn);
    in practice that produced titles like "All 1861 tests pass" or
    "Total cprofile time: 43.6ms vs 61.1ms baseline → 28%…" because the
    model's last sentence wasn't reliably a *description of what changed*.
    The fixed four-segment form is uglier but consistently informative,
    and the body's "Bottlenecks and Changes" section already provides the
    human-readable summary. The trailing segment is the model's short name
    (the part after any ``provider/`` prefix) so PRs are attributable at a
    glance.
    """
    test_name = _short_test_name(extraction)
    model_name = _short_model_name(model)
    _metric_label, delta_pct = _primary_total_delta(loop_result, survivors)
    if delta_pct is None or not survivors:
        if review_attempts:
            # Tested-but-unaccepted attempt published for review. The loop's
            # final-vs-baseline delta is zero (the attempt was reverted), so
            # we report what the attempt itself measured — that's the signed
            # number a reviewer cares about ("the model tried this and got X").
            attempt_delta = _best_attempt_delta(review_attempts)
            if attempt_delta is not None:
                return (
                    f"spinwright / {test_name} / review "
                    f"{_format_delta(attempt_delta)} / {model_name}"
                )
            return (
                f"spinwright / {test_name} / review (no measurable change) "
                f"/ {model_name}"
            )
        return f"spinwright / {test_name} / no improvements / {model_name}"
    return f"spinwright / {test_name} / {_format_delta(delta_pct)} / {model_name}"


def _short_model_name(model: str) -> str:
    """``ollama/kimi-k2.7-code`` → ``kimi-k2.7-code``; a bare name is returned
    unchanged. Only the final ``provider/`` prefix is stripped so nested paths
    keep their trailing component."""
    return model.rsplit("/", 1)[-1] or model


def _best_attempt(
    attempts: tuple[OptimizationResult, ...],
) -> OptimizationResult | None:
    """The attempt with the largest primary-metric reduction — the one whose
    number the title reports and whose measurements the table should mirror.
    Attempts without a measured ``relative_improvement`` are ignored."""
    measured = [a for a in attempts if a.relative_improvement is not None]
    if not measured:
        return None
    return max(measured, key=lambda a: a.relative_improvement)


def _best_attempt_delta(attempts: tuple[OptimizationResult, ...]) -> float | None:
    """Largest reduction across the attempts' own per-iteration measurements
    (``relative_improvement`` is the primary-metric delta the gate considered)."""
    best = _best_attempt(attempts)
    return best.relative_improvement if best is not None else None


def _format_delta(pct: float) -> str:
    """Render a relative improvement as a signed bare percentage —
    ``-51%`` for a 51% reduction, ``+12%`` for a regression. A literal zero
    is rendered as ``+0%`` (not ``-0%``) because IEEE-754 signed-zero leaks
    through ``-0.0`` otherwise."""
    signed = -pct if pct != 0 else 0.0
    return f"{signed:+.0%}"


def _short_test_name(extraction: ExtractionMetadata) -> str:
    if extraction.original_nodeid:
        return extraction.original_nodeid.rsplit("::", 1)[-1]
    return extraction.extraction_path.stem


def _module_from_path(path: str) -> str:
    """``static_frame/core/index.py`` → ``static_frame.core.index``.
    Strips a leading ``a/`` or ``b/`` that git diffs include."""
    p = path.removeprefix("a/").removeprefix("b/")
    parts = Path(p).with_suffix("").parts
    if not parts:
        return "optimization"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return "optimization"
    return ".".join(parts)


def diff_rel_paths(diff: str) -> list[str]:
    return [
        line.split()[2][2:]  # "diff --git a/X b/X" → "X"
        for line in diff.splitlines()
        if line.startswith("diff --git ")
        and len(line.split()) >= 4
        and line.split()[2].startswith("a/")
    ]


# ---------------------------------------------------------------------------
# Primary metric delta
# ---------------------------------------------------------------------------


def _primary_total_delta(
    loop_result: LoopResult, survivors: list[OptimizationResult]
) -> tuple[str, float | None]:
    """Returns (gate_metric, total_relative_improvement). The "total" is the
    delta from the *original* baseline to the *current head*. If a patch was
    dropped during regression we don't re-measure here, so this overstates
    improvement; the caller can flag it in the body."""
    if not loop_result.baseline:
        return "none", None
    if not survivors:
        return "none", None
    use_cg = (
        loop_result.baseline.callgrind is not None
        and loop_result.final_callgrind is not None
    )
    if use_cg:
        base = loop_result.baseline.callgrind.instructions
        final = loop_result.final_callgrind.instructions
        if base > 0:
            return "callgrind_instructions", (base - final) / base
    base_wt = loop_result.baseline.walltime.median_seconds
    if loop_result.final_walltime is not None and base_wt > 0:
        final_wt = loop_result.final_walltime.median_seconds
        return "walltime_median", (base_wt - final_wt) / base_wt
    return "none", None


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


def _build_body(
    *,
    loop_result: LoopResult,
    survivors: list[OptimizationResult],
    regression: RegressionResult | None,
    extraction: ExtractionMetadata,
    run_id: str,
    model: str,
    repo_dir: Path,
    review_attempts: tuple[OptimizationResult, ...] = (),
) -> str:
    metric_name, total_delta = _primary_total_delta(loop_result, survivors)
    lines: list[str] = []

    # Summary
    lines.append("## Summary")
    lines.append("")
    if survivors:
        delta_str = f"**{total_delta:.0%}**" if total_delta is not None else "(unknown)"
        metric_text = (
            "Callgrind instruction count"
            if metric_name == "callgrind_instructions"
            else "median wallclock"
        )
        modules = sorted(
            {_module_from_path(p) for it in survivors for p in diff_rel_paths(it.diff)}
        )
        module_str = ", ".join(f"`{m}`" for m in modules) or "the target package"
        test_name = extraction.original_nodeid or extraction.extraction_path.name
        lines.append(
            f"Spinwright identified and applied **{len(survivors)}** "
            f"optimization{'s' if len(survivors) != 1 else ''} to {module_str}, "
            f"reducing {metric_text} by {delta_str} on the extracted test "
            f"`{test_name}`."
        )
        if regression and regression.dropped_commits:
            lines.append("")
            lines.append(
                f"_(The agent loop accepted {len(loop_result.accepted_indices)} patches, "
                f"but the full pytest suite caught regressions in "
                f"{len(regression.dropped_commits)} of them; those were dropped "
                f"via `{regression.fallback_used}` before this PR was assembled.)_"
            )
    elif review_attempts:
        test_name = extraction.original_nodeid or extraction.extraction_path.name
        lines.append(
            "Spinwright found no change clearing the gate threshold. Published "
            "under `--always-publish`: this PR carries the model's most-improved "
            f"**tested-but-unaccepted** attempt on `{test_name}` so the change and "
            "its reasoning can be reviewed. **Not for merge as-is.**"
        )
    else:
        lines.append("Spinwright found no improvements clearing the gate threshold.")
        lines.append("This PR is informational only.")

    # Measurements table. For an accepted change we show baseline vs the final
    # on-disk state. For a reverted review attempt the final state *is* the
    # baseline (nothing was kept), so a baseline-vs-final table would render an
    # impossible all-zero delta that contradicts the title; instead we show the
    # attempt's own before/after — the same numbers the title's delta is drawn
    # from.
    lines.append("")
    lines.append("## Measurements")
    lines.append("")
    base = loop_result.baseline
    best = _best_attempt(review_attempts) if review_attempts else None
    if best is not None:
        table = _measurements_table(
            baseline_wt=best.baseline_walltime,
            final_wt=best.candidate_walltime,
            baseline_cg=best.baseline_callgrind,
            final_cg=best.candidate_callgrind,
            attempt_reverted=True,
        )
    else:
        table = _measurements_table(
            baseline_wt=base.walltime if base is not None else None,
            final_wt=loop_result.final_walltime,
            baseline_cg=base.callgrind if base is not None else None,
            final_cg=loop_result.final_callgrind,
        )
    lines.append(table)

    # Test
    lines.append("")
    lines.append("## Test")
    lines.append("")
    rel_extraction = _relative_to(extraction.extraction_path, repo_dir)
    lines.append(
        f"Extraction harness: `{rel_extraction or extraction.extraction_path}`"
    )
    if extraction.original_nodeid:
        lines.append(f"Derived from: `{extraction.original_nodeid}`")
    lines.append(f"Source commit: `{extraction.source_commit_sha}`")
    if regression:
        passed_str = "all green ✓" if regression.passed else "FAILED"
        lines.append(f"Full pytest suite after applied patches: **{passed_str}**.")

    # Bottlenecks and Changes
    lines.append("")
    lines.append("## Bottlenecks and Changes")
    if survivors:
        for i, it in enumerate(survivors, start=1):
            lines.extend(_bottleneck_section(i, it, total_metric=metric_name))
    elif review_attempts:
        for i, it in enumerate(review_attempts, start=1):
            lines.extend(_bottleneck_section(i, it, total_metric=metric_name))
            reason = (it.rejection_reason or "below gate threshold").splitlines()[0]
            lines.append("")
            lines.append(f"**Not accepted:** {reason}")
    else:
        lines.append("")
        lines.append("None accepted.")

    # Dropped patches (regression detail)
    if regression and regression.dropped_commits:
        lines.append("")
        lines.append("## Dropped patches")
        lines.append("")
        lines.append(
            f"The following commits were initially accepted by the agent loop "
            f"but reverted by the regression check (`{regression.fallback_used}`):"
        )
        lines.append("")
        for sha in regression.dropped_commits:
            lines.append(f"- `{sha[:12]}`")

    # Notes
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"Generated by Spinwright v{spinwright.__version__} using {model}.")
    lines.append(f"Run ID: `{run_id}`.")
    return "\n".join(lines) + "\n"


def _measurements_table(
    *,
    baseline_wt: WalltimeResult | None,
    final_wt: WalltimeResult | None,
    baseline_cg: CallgrindResult | None,
    final_cg: CallgrindResult | None,
    attempt_reverted: bool = False,
) -> str:
    """Render a baseline-vs-after table. ``attempt_reverted`` labels the "After"
    column as a tested-but-reverted attempt (the numbers the model measured with
    the patch applied, not the current on-disk state)."""
    rows: list[tuple[str, str, str, str]] = []
    after_label = "After (reverted attempt)" if attempt_reverted else "After"
    if baseline_cg is not None and final_cg is not None:
        base = baseline_cg.instructions
        final = final_cg.instructions
        delta = (base - final) / base if base > 0 else 0
        rows.append(
            (
                "Callgrind instructions (per call)",
                f"{base:,}",
                f"{final:,}",
                f"−{delta:.1%}",
            )
        )
    if baseline_wt is not None and final_wt is not None:
        base_med = baseline_wt.median_seconds
        final_med = final_wt.median_seconds
        delta = (base_med - final_med) / base_med if base_med > 0 else 0
        rows.append(
            (
                "Wallclock median (µs)",
                f"{base_med * 1e6:.2f}",
                f"{final_med * 1e6:.2f}",
                f"−{delta:.1%}",
            )
        )
        rows.append(
            (
                "Wallclock best (µs)",
                f"{baseline_wt.best_seconds * 1e6:.2f}",
                f"{final_wt.best_seconds * 1e6:.2f}",
                "",
            )
        )
        rows.append(
            (
                "Wallclock stddev (µs)",
                f"{baseline_wt.stddev_seconds * 1e6:.2f}",
                f"{final_wt.stddev_seconds * 1e6:.2f}",
                "",
            )
        )
    if not rows:
        return "_(no measurements recorded)_"
    header = f"| Metric | Baseline | {after_label} | Δ |"
    sep = "|---|---|---|---|"
    body = "\n".join(f"| {a} | {b_} | {c} | {d} |" for a, b_, c, d in rows)
    note = ""
    if final_wt is not None:
        note = (
            f"\n\nWallclock measured over {final_wt.repeats} repeats "
            f"of {final_wt.iterations_per_repeat} iterations each."
        )
    if attempt_reverted:
        note += (
            "\n\n_“After” reflects the tested-but-unaccepted attempt measured "
            "with the patch applied; it was reverted, so the current tree matches "
            "“Baseline”._"
        )
    return f"{header}\n{sep}\n{body}{note}"


def _bottleneck_section(
    index: int, it: OptimizationResult, *, total_metric: str
) -> list[str]:
    lines: list[str] = []
    file_hint = ""
    if it.conversation is not None:
        # Pull funcname from the focus hint embedded in the user message;
        # fall back to "the modified function" if not parseable.
        pass
    # Try to find the focus from the diff
    diff_paths = diff_rel_paths(it.diff)
    file_hint = diff_paths[0] if diff_paths else "the modified file"
    summary = ""
    if it.conversation is not None:
        summary = (it.conversation.final_text or "").strip()
    if not summary:
        summary = "(no summary provided)"
    primary_delta = it.relative_improvement
    primary_text = f"{primary_delta:+.1%}" if primary_delta is not None else "n/a"

    lines.append("")
    lines.append(f"### Patch {index}: `{file_hint}`")
    lines.append("")
    lines.append(f"**Summary:** {summary}")
    lines.append("")
    lines.append(f"**Local impact:** {primary_text} on {it.gate_metric}")
    if it.relative_walltime_improvement is not None:
        lines.append(f"  - walltime: {it.relative_walltime_improvement:+.1%}")
    if it.relative_callgrind_improvement is not None:
        lines.append(f"  - callgrind: {it.relative_callgrind_improvement:+.1%}")
    if it.commit_sha:
        lines.append(f"**Commit:** `{it.commit_sha[:12]}`")
    if it.diff:
        lines.append("")
        lines.append("```diff")
        # Keep diff bounded; long diffs are unfriendly to PR readers.
        diff_lines = it.diff.splitlines()
        if len(diff_lines) > 60:
            lines.extend(diff_lines[:60])
            lines.append(f"... ({len(diff_lines) - 60} more lines truncated)")
        else:
            lines.extend(diff_lines)
        lines.append("```")
    return lines


def _relative_to(path: Path, base: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return None
