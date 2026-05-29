from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from spinwright.config import Config
from spinwright.llm.client import ClientProtocol
from spinwright.measurement.runner import DriverError
from spinwright.measurement.types import CallgrindResult, VerifyResult, WalltimeResult
from spinwright.optimization.optimize import (
    FocusHint,
    OptimizationResult,
    _dual_measure,
    optimize_once,
)
from spinwright.profiling import cprofile
from spinwright.repo import venv as venv_mod
from spinwright.repo.workspace import Workspace


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopBaseline:
    walltime: WalltimeResult
    callgrind: CallgrindResult | None
    callgrind_disabled_reason: str | None
    verify: VerifyResult


@dataclass
class LoopResult:
    success: bool
    extraction_path: Path
    baseline: LoopBaseline | None
    iterations: list[OptimizationResult]
    accepted_indices: list[int]
    final_walltime: WalltimeResult | None
    final_callgrind: CallgrindResult | None
    explored: list[str]
    stop_reason: str
    failure_reason: str | None = None

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_indices)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def run_loop(
    *,
    ws: Workspace,
    extraction_path: Path,
    config: Config,
    client: ClientProtocol,
    model: str | None = None,
    extra_excludes: tuple[str, ...] = (),
) -> LoopResult:
    """Iterate optimize_once until budgets exhaust or no more candidates.

    Per-iteration loop:
      1. Profile current state via cProfile (in the target venv).
      2. Pick the hottest user-code function (under ``ws.repo_dir``) whose
         explored_key isn't in the ``explored`` set yet.
      3. Run ``optimize_once`` with that focus_hint.
      4. If accepted: record, refresh baseline to the candidate's measurements,
         continue.
      5. If rejected: add the focus to ``explored``, continue.

    Stops when:
      - No candidate survives the explored filter.
      - ``budget.max_patches_proposed`` iterations have been attempted.
      - The candidate profile fails (e.g., user-edit broke import) — captured
         as ``stop_reason="infrastructure_error"``.
    """
    venv_python = venv_mod.python_executable(ws)
    repo_dir_resolved = ws.repo_dir.resolve()

    base_wt, base_vr, base_cg, cg_disabled = _dual_measure(
        venv_python, extraction_path,
        repeats=config.measurement.walltime_repeats,
        cwd=ws.repo_dir,
        config=config,
    )
    baseline = LoopBaseline(
        walltime=base_wt, callgrind=base_cg,
        callgrind_disabled_reason=cg_disabled, verify=base_vr,
    )
    if not base_vr.passed:
        return LoopResult(
            success=False,
            extraction_path=extraction_path,
            baseline=baseline,
            iterations=[],
            accepted_indices=[],
            final_walltime=base_wt,
            final_callgrind=base_cg,
            explored=[],
            stop_reason="baseline_verify_failed",
            failure_reason="baseline verify() failed before any iteration ran",
        )

    iterations: list[OptimizationResult] = []
    accepted_indices: list[int] = []
    explored: list[str] = []
    current_wt = base_wt
    current_cg = base_cg
    max_iters = config.budget.max_patches_proposed
    stop_reason = "max_iterations"

    for i in range(max_iters):
        try:
            prof = cprofile.profile_cprofile(
                venv_python, extraction_path,
                iterations=200,
                cwd=ws.repo_dir,
            )
        except DriverError as e:
            stop_reason = "infrastructure_error"
            iterations.append(_failed_iteration(extraction_path, current_wt, current_cg,
                                                f"cProfile driver failed: {e}"))
            break

        focus = _select_focus(
            prof, repo_dir_resolved, explored,
            corpus_dir=config.corpus.dir,
        )
        if focus is None:
            stop_reason = "no_remaining_bottlenecks"
            break

        result = optimize_once(
            ws=ws,
            extraction_path=extraction_path,
            config=config,
            client=client,
            model=model,
            extra_excludes=extra_excludes,
            focus_hint=focus,
        )
        iterations.append(result)

        if result.accepted:
            accepted_indices.append(i)
            if result.candidate_walltime is not None:
                current_wt = result.candidate_walltime
            if result.candidate_callgrind is not None:
                current_cg = result.candidate_callgrind
        else:
            explored.append(focus.explored_key)

    return LoopResult(
        success=True,
        extraction_path=extraction_path,
        baseline=baseline,
        iterations=iterations,
        accepted_indices=accepted_indices,
        final_walltime=current_wt,
        final_callgrind=current_cg,
        explored=explored,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Focus selection
# ---------------------------------------------------------------------------


_FUNCNAME_BLACKLIST = {"<module>", "<genexpr>", "<listcomp>", "<setcomp>",
                       "<dictcomp>", "<lambda>"}


def _select_focus(
    prof: cprofile.CProfileResult,
    repo_dir: Path,
    explored: list[str],
    *,
    corpus_dir: str = "spinwright",
) -> FocusHint | None:
    """Pick the top-cumtime entry that lives in the repo dir and hasn't been
    explored yet. Skips Python-internal funcnames and the extraction harness
    itself (the LLM should optimize the *target package*, not the harness).

    ``corpus_dir`` is the repo-relative path where extractions live; it's
    used as a substring match so any nested layout under that root counts
    as harness too (e.g. ``spinwright/runs/...``).
    """
    repo_str = str(repo_dir)
    explored_set = set(explored)
    # Resolve corpus to an absolute prefix and a substring fragment; either
    # match disqualifies the entry. (Absolute is for the normal case;
    # substring is a defensive fallback for path-canonicalisation quirks.)
    corpus_abs = str((repo_dir / corpus_dir).resolve())
    corpus_substr = corpus_dir.strip("/").strip()

    ranked = sorted(prof.entries, key=lambda e: e.cumtime, reverse=True)
    for entry in ranked:
        if not entry.filename or not entry.filename.startswith(repo_str):
            continue
        if entry.funcname in _FUNCNAME_BLACKLIST:
            continue
        if entry.filename.startswith(corpus_abs) or (
            corpus_substr and corpus_substr in entry.filename
        ):
            # The extraction harness itself; not what we want to optimize.
            continue
        key = _explored_key(entry.filename, entry.lineno, entry.funcname)
        if key in explored_set:
            continue
        return FocusHint(
            funcname=entry.funcname,
            filename=entry.filename,
            lineno=entry.lineno,
            qualname=_guess_qualname(entry.filename, entry.funcname, repo_dir),
            cumtime_seconds=entry.cumtime,
            explored_key=key,
        )
    return None


def _explored_key(filename: str, lineno: int, funcname: str) -> str:
    return f"{filename}:{lineno}:{funcname}"


def _guess_qualname(filename: str, funcname: str, repo_dir: Path) -> str | None:
    """Build a dotted import path from a filename, best-effort.

    E.g., ``/repo/static_frame/core/index.py`` + ``_loc_to_iloc`` →
    ``static_frame.core.index._loc_to_iloc``. Returns None if the path
    structure doesn't look like a regular package layout.
    """
    try:
        rel = Path(filename).resolve().relative_to(repo_dir)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if not parts:
        return None
    # Drop a trailing __init__ — `pkg/__init__.py` is `pkg`, not `pkg.__init__`.
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return None
    return ".".join(parts + [funcname])


def _failed_iteration(
    extraction_path: Path,
    wt: WalltimeResult | None,
    cg: CallgrindResult | None,
    reason: str,
) -> OptimizationResult:
    return OptimizationResult(
        accepted=False,
        nodeid_or_extraction=str(extraction_path),
        baseline_walltime=wt,
        candidate_walltime=None,
        baseline_callgrind=cg,
        candidate_callgrind=None,
        candidate_verify=None,
        relative_improvement=None,
        relative_walltime_improvement=None,
        relative_callgrind_improvement=None,
        gate_metric="none",
        threshold=0.0,
        diff="",
        commit_sha=None,
        rejection_reason=reason,
        conversation=None,
        extraction_path=extraction_path,
    )
