from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from spinwright.config import Config
from spinwright.llm.client import ClientProtocol
from spinwright.llm.dispatch import ConversationResult, run_conversation
from spinwright.measurement import walltime
from spinwright.measurement.types import VerifyResult, WalltimeResult
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
    candidate_verify: VerifyResult | None
    relative_improvement: float | None      # (baseline - candidate) / baseline, on median
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

    baseline_wt, baseline_vr = walltime.measure(
        venv_python, extraction_path, repeats=repeats, cwd=ws.repo_dir,
    )
    if not baseline_vr.passed:
        return OptimizationResult(
            accepted=False,
            nodeid_or_extraction=str(extraction_path),
            baseline_walltime=baseline_wt,
            candidate_walltime=None,
            candidate_verify=baseline_vr,
            relative_improvement=None,
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
    system_prompt = _build_system_prompt(threshold=threshold, extra_excludes=extra_excludes)
    user_message = _build_user_message(
        extraction_path=extraction_path,
        baseline_wt=baseline_wt,
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
            candidate_verify=None,
            relative_improvement=None,
            threshold=threshold,
            diff="",
            commit_sha=None,
            rejection_reason=f"conversation ended ({conversation.stop_reason}) without applying any edits",
            conversation=conversation,
            extraction_path=extraction_path,
        )

    candidate_wt, candidate_vr = walltime.measure(
        venv_python, extraction_path, repeats=repeats, cwd=ws.repo_dir,
    )
    rel_improve = _relative_improvement(baseline_wt, candidate_wt)

    if not candidate_vr.passed:
        revert = git.git_revert_all(ws.repo_dir)
        return OptimizationResult(
            accepted=False,
            nodeid_or_extraction=str(extraction_path),
            baseline_walltime=baseline_wt,
            candidate_walltime=candidate_wt,
            candidate_verify=candidate_vr,
            relative_improvement=rel_improve,
            threshold=threshold,
            diff=diff,
            commit_sha=None,
            rejection_reason="candidate verify() failed",
            conversation=conversation,
            extraction_path=extraction_path,
            reverted_paths=revert.reverted_paths,
        )

    if rel_improve is None or rel_improve < threshold:
        revert = git.git_revert_all(ws.repo_dir)
        reason = (
            f"improvement {rel_improve:.2%} is below threshold {threshold:.0%}"
            if rel_improve is not None else "could not compute improvement"
        )
        return OptimizationResult(
            accepted=False,
            nodeid_or_extraction=str(extraction_path),
            baseline_walltime=baseline_wt,
            candidate_walltime=candidate_wt,
            candidate_verify=candidate_vr,
            relative_improvement=rel_improve,
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
        f"(-{rel_improve:.0%} median walltime)"
    )
    sha = workspace_mod.commit(ws, changed_paths, commit_msg)
    return OptimizationResult(
        accepted=True,
        nodeid_or_extraction=str(extraction_path),
        baseline_walltime=baseline_wt,
        candidate_walltime=candidate_wt,
        candidate_verify=candidate_vr,
        relative_improvement=rel_improve,
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


def _relative_improvement(
    baseline: WalltimeResult, candidate: WalltimeResult
) -> float | None:
    if baseline.median_seconds <= 0:
        return None
    return (baseline.median_seconds - candidate.median_seconds) / baseline.median_seconds


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

The orchestrator measures wallclock around `run(state)` before and after \
your changes (best-of-K with timeit.autorange). To be accepted, your patch \
must achieve a median wallclock reduction of at least {threshold:.0%} \
without breaking verify().

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
    *, threshold: float, extra_excludes: tuple[str, ...]
) -> str:
    base = _SYSTEM_PROMPT_TEMPLATE.format(threshold=threshold)
    if extra_excludes:
        base += "\nSuggested profile excludes for this repo: " + ", ".join(
            f"`{e}`" for e in extra_excludes
        )
    return base


def _build_user_message(
    *, extraction_path: Path, baseline_wt: WalltimeResult
) -> str:
    return (
        f"Extraction to optimize: `{extraction_path}`\n\n"
        f"Baseline wallclock (best of {baseline_wt.repeats}, "
        f"{baseline_wt.iterations_per_repeat} iters/repeat):\n"
        f"- best:   {baseline_wt.best_seconds * 1e6:.2f} us\n"
        f"- median: {baseline_wt.median_seconds * 1e6:.2f} us\n"
        f"- stddev: {baseline_wt.stddev_seconds * 1e6:.2f} us\n\n"
        "Start by profiling. Identify the hottest function in the target "
        "package's own code. Read its source, propose ONE edit, sanity-check "
        "it, and end the turn."
    )
