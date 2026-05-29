from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from spinwright import platform as platform_mod
from spinwright.config import Config
from spinwright.llm.client import ClientProtocol
from spinwright.llm.dispatch import ConversationResult, run_conversation
from spinwright.measurement import callgrind as callgrind_mod
from spinwright.measurement import walltime
from spinwright.measurement.types import CallgrindResult, VerifyResult, WalltimeResult
from spinwright.repo import venv as venv_mod
from spinwright.repo import workspace as workspace_mod
from spinwright.repo.workspace import Workspace
from spinwright.tools import git, registry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    accepted: bool
    nodeid_or_extraction: str
    baseline_walltime: WalltimeResult | None
    candidate_walltime: WalltimeResult | None
    baseline_callgrind: CallgrindResult | None
    candidate_callgrind: CallgrindResult | None
    candidate_verify: VerifyResult | None
    relative_improvement: float | None      # primary metric (see gate_metric)
    relative_walltime_improvement: float | None
    relative_callgrind_improvement: float | None
    gate_metric: str                        # "callgrind_instructions" | "walltime_median" | "none"
    threshold: float
    diff: str
    commit_sha: str | None
    rejection_reason: str | None
    conversation: ConversationResult | None
    extraction_path: Path
    reverted_paths: list[str] = field(default_factory=list)


def optimize_once(
    *,
    ws: Workspace,
    extraction_path: Path,
    config: Config,
    client: ClientProtocol,
    model: str | None = None,
    extra_excludes: tuple[str, ...] = (),
) -> OptimizationResult:
    """Run one round of the optimization agent against ``extraction_path``.

    Workflow:
      1. Measure baseline wallclock + verify.
      2. Drive the LLM optimization conversation (profile, propose, edit, sanity).
      3. Measure candidate wallclock + verify.
      4. If verify passes AND median-walltime improvement >= threshold: commit
         the LLM's edits on the working branch.
      5. Otherwise: revert all tracked changes and report.

    The baseline measurement is always taken before the LLM is invoked so
    the LLM's view of "what counts as improvement" matches the orchestrator's.
    """
    venv_python = venv_mod.python_executable(ws)
    threshold = config.measurement.improvement_threshold
    repeats = config.measurement.walltime_repeats

    baseline_wt, baseline_vr, baseline_cg, callgrind_disabled_reason = _dual_measure(
        venv_python, extraction_path, repeats=repeats, cwd=ws.repo_dir, config=config,
    )
    if not baseline_vr.passed:
        return OptimizationResult(
            accepted=False,
            nodeid_or_extraction=str(extraction_path),
            baseline_walltime=baseline_wt,
            candidate_walltime=None,
            baseline_callgrind=baseline_cg,
            candidate_callgrind=None,
            candidate_verify=baseline_vr,
            relative_improvement=None,
            relative_walltime_improvement=None,
            relative_callgrind_improvement=None,
            gate_metric="none",
            threshold=threshold,
            diff="",
            commit_sha=None,
            rejection_reason="baseline verify() failed before any edits were made",
            conversation=None,
            extraction_path=extraction_path,
        )

    tools = registry.build_optimization_tools(
        workspace_root=ws.root,
        repo_dir=ws.repo_dir,
        venv_python=venv_python,
        extraction_path=extraction_path,
        profile_default_iterations=config.measurement.walltime_repeats * 200,
    )
    primary_metric = "callgrind_instructions" if baseline_cg is not None else "walltime_median"
    system_prompt = _build_system_prompt(
        threshold=threshold,
        extra_excludes=extra_excludes,
        primary_metric=primary_metric,
    )
    user_message = _build_user_message(
        extraction_path=extraction_path,
        baseline_wt=baseline_wt,
        baseline_cg=baseline_cg,
        callgrind_disabled_reason=callgrind_disabled_reason,
    )
    chosen_model = model or config.models.reasoning

    conversation = run_conversation(
        client,
        model=chosen_model,
        system=system_prompt,
        initial_user_message=user_message,
        tools=tools,
        max_turns=config.budget.max_extraction_turns,  # reuse extraction turn cap for now
    )

    diff = git.git_diff(ws.repo_dir)

    if not diff.strip():
        return OptimizationResult(
            accepted=False,
            nodeid_or_extraction=str(extraction_path),
            baseline_walltime=baseline_wt,
            candidate_walltime=None,
            baseline_callgrind=baseline_cg,
            candidate_callgrind=None,
            candidate_verify=None,
            relative_improvement=None,
            relative_walltime_improvement=None,
            relative_callgrind_improvement=None,
            gate_metric=primary_metric,
            threshold=threshold,
            diff="",
            commit_sha=None,
            rejection_reason=f"conversation ended ({conversation.stop_reason}) without applying any edits",
            conversation=conversation,
            extraction_path=extraction_path,
        )

    candidate_wt, candidate_vr, candidate_cg, _ = _dual_measure(
        venv_python, extraction_path, repeats=repeats, cwd=ws.repo_dir,
        config=config, callgrind_enabled=(baseline_cg is not None),
    )
    rel_wt = _relative_walltime_improvement(baseline_wt, candidate_wt)
    rel_cg = _relative_callgrind_improvement(baseline_cg, candidate_cg)
    rel_primary = rel_cg if primary_metric == "callgrind_instructions" else rel_wt
    gate_metric = primary_metric

    if not candidate_vr.passed:
        revert = git.git_revert_all(ws.repo_dir)
        return OptimizationResult(
            accepted=False,
            nodeid_or_extraction=str(extraction_path),
            baseline_walltime=baseline_wt,
            candidate_walltime=candidate_wt,
            baseline_callgrind=baseline_cg,
            candidate_callgrind=candidate_cg,
            candidate_verify=candidate_vr,
            relative_improvement=rel_primary,
            relative_walltime_improvement=rel_wt,
            relative_callgrind_improvement=rel_cg,
            gate_metric=gate_metric,
            threshold=threshold,
            diff=diff,
            commit_sha=None,
            rejection_reason="candidate verify() failed",
            conversation=conversation,
            extraction_path=extraction_path,
            reverted_paths=revert.reverted_paths,
        )

    if rel_primary is None or rel_primary < threshold:
        revert = git.git_revert_all(ws.repo_dir)
        reason = (
            f"{_metric_label(gate_metric)} improvement {rel_primary:.2%} is below threshold {threshold:.0%}"
            if rel_primary is not None else "could not compute improvement"
        )
        return OptimizationResult(
            accepted=False,
            nodeid_or_extraction=str(extraction_path),
            baseline_walltime=baseline_wt,
            candidate_walltime=candidate_wt,
            baseline_callgrind=baseline_cg,
            candidate_callgrind=candidate_cg,
            candidate_verify=candidate_vr,
            relative_improvement=rel_primary,
            relative_walltime_improvement=rel_wt,
            relative_callgrind_improvement=rel_cg,
            gate_metric=gate_metric,
            threshold=threshold,
            diff=diff,
            commit_sha=None,
            rejection_reason=reason,
            conversation=conversation,
            extraction_path=extraction_path,
            reverted_paths=revert.reverted_paths,
        )

    # Accept: figure out which files changed, commit.
    changed_paths = _diff_paths(ws.repo_dir, diff)
    commit_msg = (
        f"spinwright: optimize {extraction_path.name} "
        f"(-{rel_primary:.0%} {_metric_label(gate_metric)})"
    )
    sha = workspace_mod.commit(ws, changed_paths, commit_msg)
    return OptimizationResult(
        accepted=True,
        nodeid_or_extraction=str(extraction_path),
        baseline_walltime=baseline_wt,
        candidate_walltime=candidate_wt,
        baseline_callgrind=baseline_cg,
        candidate_callgrind=candidate_cg,
        candidate_verify=candidate_vr,
        relative_improvement=rel_primary,
        relative_walltime_improvement=rel_wt,
        relative_callgrind_improvement=rel_cg,
        gate_metric=gate_metric,
        threshold=threshold,
        diff=diff,
        commit_sha=sha,
        rejection_reason=None,
        conversation=conversation,
        extraction_path=extraction_path,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dual_measure(
    venv_python: Path,
    extraction_path: Path,
    *,
    repeats: int,
    cwd: Path,
    config: Config,
    callgrind_enabled: bool | None = None,
) -> tuple[WalltimeResult, VerifyResult, CallgrindResult | None, str | None]:
    """Run walltime + (Linux) Callgrind on the same extraction. Returns
    (walltime, verify, callgrind_or_none, callgrind_disabled_reason).

    ``callgrind_enabled``: tri-state. None → "try if Linux"; True → "must try
    (used for candidate after baseline succeeded)"; False → skip entirely.
    """
    wt, vr = walltime.measure(
        venv_python, extraction_path, repeats=repeats, cwd=cwd,
    )
    if not vr.passed or callgrind_enabled is False:
        return wt, vr, None, None
    if callgrind_enabled is None and not platform_mod.is_linux():
        return wt, vr, None, "macOS — Linux-only metric"
    try:
        cg, cg_vr = callgrind_mod.measure_callgrind(
            venv_python, extraction_path,
            valgrind_path=config.measurement.callgrind_path,
            autoscale_min_instructions=config.measurement.autoscale_min_instructions,
            cwd=cwd,
        )
    except callgrind_mod.CallgrindUnavailable as e:
        return wt, vr, None, str(e)
    # If callgrind's verify disagrees (rare — would indicate flaky verify), prefer
    # the strict failure so the caller sees the discrepancy.
    if not cg_vr.passed:
        return wt, cg_vr, None, "callgrind run failed verify"
    return wt, vr, cg, None


def _relative_walltime_improvement(
    baseline: WalltimeResult | None, candidate: WalltimeResult | None
) -> float | None:
    if baseline is None or candidate is None or baseline.median_seconds <= 0:
        return None
    return (baseline.median_seconds - candidate.median_seconds) / baseline.median_seconds


def _relative_callgrind_improvement(
    baseline: CallgrindResult | None, candidate: CallgrindResult | None
) -> float | None:
    if baseline is None or candidate is None or baseline.instructions <= 0:
        return None
    return (baseline.instructions - candidate.instructions) / baseline.instructions


# Backwards-compat alias for tests written against M2.3.
_relative_improvement = _relative_walltime_improvement


def _metric_label(gate_metric: str) -> str:
    return {
        "callgrind_instructions": "Callgrind instructions",
        "walltime_median": "median walltime",
        "none": "(no metric)",
    }.get(gate_metric, gate_metric)


def _diff_paths(repo_dir: Path, diff: str) -> list[Path]:
    """Pull the paths mentioned in `git diff HEAD` lines like ``diff --git a/X b/X``."""
    paths: list[Path] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            # "diff --git a/<path> b/<path>"
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/"):
                rel = parts[2][2:]
                paths.append((repo_dir / rel).resolve())
    return paths


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE = """\
You are Spinwright's optimization agent. You have ONE shot at proposing a
patch that makes the extraction's `run()` faster, with the verify() check
still passing.

The orchestrator measures `run(state)` before and after your changes. The
primary metric on this run is {primary_metric_label}; to be accepted, your
patch must reduce it by at least {threshold:.0%} without breaking verify().
({callgrind_note})

Tools available:
- `profile_cprofile` — get hot functions. Default sort: cumtime. Use
  exclude_paths to drop stdlib/third-party noise once you've identified
  the user-code hotspots.
- `read_source` — look at the source of any dotted qualname.
- `edit_file` — string-replace edit in a workspace file. old_string must
  match exactly once; include surrounding context if needed.
- `write_file` — overwrite a workspace file. Avoid unless rewriting.
- `run_python` — sanity-check your edit by importing the extraction and
  running setup → run → verify once.
- `git_diff` — see your current uncommitted changes.
- `git_revert_path` / `git_revert_all` — undo if you change your mind.

Rules:
- Pure-Python edits only. No new C extensions, no Cython, no Rust. Swapping
  a pure-Python implementation for an already-available stdlib/NumPy/SciPy
  equivalent IS allowed.
- Do not modify the test's public API or the extraction's `setup`/`run`/
  `verify` signatures.
- Do not weaken or change `verify()`. If your edit makes verify fail, you
  need a different edit, not a different verify.
- Sanity-check with run_python before ending the turn. If sanity fails,
  either fix it or revert before ending the turn.
- One round: end the turn after you're confident in your change OR after
  you've decided no improvement is available (in which case explicitly say
  so and call git_revert_all to leave the tree clean).

End the turn with a brief one-sentence summary of what you changed and why.
"""


def _build_system_prompt(
    *, threshold: float, extra_excludes: tuple[str, ...], primary_metric: str
) -> str:
    label = _metric_label(primary_metric)
    note = (
        "Callgrind is the canonical gate (deterministic across runs); "
        "median wallclock is reported as sanity."
        if primary_metric == "callgrind_instructions"
        else "Callgrind isn't available here (macOS or missing valgrind); "
             "median wallclock with timeit.autorange best-of-K is the gate."
    )
    base = _SYSTEM_PROMPT_TEMPLATE.format(
        threshold=threshold,
        primary_metric_label=label,
        callgrind_note=note,
    )
    if extra_excludes:
        base += "\nSuggested profile excludes for this repo: " + ", ".join(
            f"`{e}`" for e in extra_excludes
        )
    return base


def _build_user_message(
    *,
    extraction_path: Path,
    baseline_wt: WalltimeResult,
    baseline_cg: CallgrindResult | None,
    callgrind_disabled_reason: str | None,
) -> str:
    lines = [
        f"Extraction to optimize: `{extraction_path}`",
        "",
        f"Baseline wallclock (best of {baseline_wt.repeats}, "
        f"{baseline_wt.iterations_per_repeat} iters/repeat):",
        f"- best:   {baseline_wt.best_seconds * 1e6:.2f} us",
        f"- median: {baseline_wt.median_seconds * 1e6:.2f} us",
        f"- stddev: {baseline_wt.stddev_seconds * 1e6:.2f} us",
    ]
    if baseline_cg is not None:
        lines.extend([
            "",
            "Baseline Callgrind (per-call instructions, two-run subtraction):",
            f"- instructions:  {baseline_cg.instructions:,}",
            f"- autoscale N:   {baseline_cg.autoscale_iterations:,}",
        ])
    elif callgrind_disabled_reason:
        lines.extend(["", f"(Callgrind disabled: {callgrind_disabled_reason})"])
    lines.extend([
        "",
        "Start by profiling. Identify the hottest function in the target "
        "package's own code. Read its source, propose ONE edit, sanity-check "
        "it, and end the turn.",
    ])
    return "\n".join(lines)
