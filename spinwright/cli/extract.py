from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from spinwright import config as cfg_mod
from spinwright.extraction import discovery, eligibility
from spinwright.extraction import extract as extract_mod
from spinwright.llm import client as client_mod
from spinwright.llm.client import ClientProtocol
from spinwright.repo import venv as venv_mod
from spinwright.repo import workspace as workspace_mod


def _load_config(path: str | None) -> cfg_mod.Config:
    if path:
        return cfg_mod.load(path)
    return cfg_mod.default()


def _looks_like_workspace(path: Path) -> bool:
    return (path / ".venv" / "bin" / "python").exists() and (path / "repo" / ".git").exists()


def _prep_fresh_workspace(
    *,
    repo_arg: str,
    cfg: cfg_mod.Config,
) -> workspace_mod.Workspace:
    print(f"prepping workspace from {repo_arg!r} ...", file=sys.stderr)
    ws = workspace_mod.create(
        source=repo_arg,
        ref=cfg.target.ref if cfg.target.ref else None,
        branch_prefix=cfg.pr.branch_prefix,
        branch_suffix="extract",
        keep=True,
    )
    print(f"  workspace: {ws.root}", file=sys.stderr)
    print(f"  base sha:  {ws.base_sha}", file=sys.stderr)
    venv_mod.create(ws)
    venv_mod.install_target(ws)
    return ws


def _resolve_workspace(
    repo_arg: str,
    *,
    cfg: cfg_mod.Config,
) -> workspace_mod.Workspace:
    """Auto-detect: if the arg points at an existing prep'd workspace, reuse
    it. Otherwise, prep a fresh one. We never delete the workspace — the user
    just paid for an LLM extraction and may want to copy the result out."""
    candidate = Path(repo_arg).expanduser()
    if candidate.exists() and _looks_like_workspace(candidate.resolve()):
        ws = workspace_mod.reuse(candidate.resolve())
        print(f"reusing workspace at {ws.root}", file=sys.stderr)
        return ws
    return _prep_fresh_workspace(repo_arg=repo_arg, cfg=cfg)


# ---------------------------------------------------------------------------
# Candidate discovery + auto-selection (SPEC §5.1)
# ---------------------------------------------------------------------------


def _scan_candidates(
    ws: workspace_mod.Workspace, cfg: cfg_mod.Config,
) -> list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]]:
    """Run pytest with --durations once, then run AST eligibility on each slow
    test. Returns the full list (eligible + ineligible) sorted by duration desc.
    """
    print("discovering slow tests (running full suite with --durations=0) ...", file=sys.stderr)
    report = discovery.discover_verbose(
        venv_python=venv_mod.python_executable(ws),
        repo_dir=ws.repo_dir,
        slow_threshold_seconds=cfg.test_selection.slow_threshold_seconds,
    )
    print(f"  pytest exit code: {report.returncode}", file=sys.stderr)
    print(f"  found {len(report.durations)} test(s) at or above "
          f"{cfg.test_selection.slow_threshold_seconds}s", file=sys.stderr)
    if not report.durations:
        # pytest exit codes: 0 = all passed, 1 = some failed, 2 = error,
        # 3 = interrupted, 4 = usage error, 5 = no tests collected.
        if report.returncode != 0:
            print("  pytest stdout tail:", file=sys.stderr)
            for line in report.stdout_tail.splitlines():
                print(f"    {line}", file=sys.stderr)
            if report.stderr_tail:
                print("  pytest stderr tail:", file=sys.stderr)
                for line in report.stderr_tail.splitlines():
                    print(f"    {line}", file=sys.stderr)
            if report.returncode == 5:
                print("  → pytest collected zero tests. Check that the test paths "
                      "exist and that any required test deps are installed.",
                      file=sys.stderr)
            else:
                print("  → discovery ran but no slow tests cleared the threshold. "
                      "Possible causes: (a) collection errored above — install "
                      "the target's test extras, e.g. `spinwright prep <repo> "
                      "--extras test,dev`; (b) threshold too high; (c) tests "
                      "live somewhere pytest's default discovery doesn't reach.",
                      file=sys.stderr)
    results: list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]] = []
    for nd in report.durations:
        test_file = ws.repo_dir / nd.nodeid.split("::")[0]
        if not test_file.exists():
            continue
        r = eligibility.check(
            test_file, nd.nodeid,
            allow_pure_conftest_imports=cfg.eligibility.allow_pure_conftest_imports,
        )
        results.append((nd, r))
    return results


def _print_candidates(
    candidates: list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]],
    *, limit: int = 25,
) -> None:
    eligible = [(nd, r) for nd, r in candidates if r.eligible]
    rejected = [(nd, r) for nd, r in candidates if not r.eligible]
    print()
    print(f"Eligible candidates ({len(eligible)}):")
    for nd, _ in eligible[:limit]:
        print(f"  {nd.seconds:7.3f}s   {nd.nodeid}")
    if len(eligible) > limit:
        print(f"  ... ({len(eligible) - limit} more)")
    print()
    print(f"Rejected candidates ({len(rejected)}):")
    for nd, r in rejected[:limit]:
        codes = ",".join(sorted({reason.code for reason in r.reasons}))
        print(f"  {nd.seconds:7.3f}s   {nd.nodeid}  [{codes}]")
    if len(rejected) > limit:
        print(f"  ... ({len(rejected) - limit} more)")


def _auto_select(
    candidates: list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]],
) -> str | None:
    eligible = [nd for nd, r in candidates if r.eligible]
    if not eligible:
        print("\nNo slow eligible tests found.", file=sys.stderr)
        print("  Pass --list-candidates to inspect the rejection reasons.", file=sys.stderr)
        return None
    chosen = eligible[0]  # discovery returns sorted descending by duration
    print(f"\nAuto-selected (highest cumtime, eligible): "
          f"{chosen.nodeid}  ({chosen.seconds:.3f}s)", file=sys.stderr)
    if len(eligible) > 1:
        runners_up = eligible[1:6]
        print(f"  ({len(eligible) - 1} more eligible candidates available; "
              f"next few: {', '.join(c.nodeid for c in runners_up)})", file=sys.stderr)
    return chosen.nodeid


def _report(result: extract_mod.ExtractionResult) -> None:
    if result.success:
        print()
        print(f"Extraction succeeded for {result.nodeid}")
        print(f"  path:    {result.extraction_path}")
        print(f"  notes:   {result.notes_path}")
        print(f"  commit:  {result.commit_sha}")
        if result.conversation:
            c = result.conversation
            print(f"  tokens:  in={c.input_tokens} out={c.output_tokens} "
                  f"cache_w={c.cache_creation_input_tokens} cache_r={c.cache_read_input_tokens}")
            print(f"  turns:   {len(c.turns)}  stop_reason={c.stop_reason}")
        return

    print()
    print(f"Extraction FAILED for {result.nodeid}")
    print(f"  reason: {result.failure_reason}")
    if result.eligibility_reasons:
        print("  eligibility reasons:")
        for r in result.eligibility_reasons:
            loc = f" (line {r.lineno})" if r.lineno is not None else ""
            print(f"    - [{r.code}] {r.message}{loc}")
    if result.sanity_error:
        print("  sanity error:")
        for line in result.sanity_error.splitlines():
            print(f"    {line}")
    if result.conversation:
        c = result.conversation
        print(f"  conversation: turns={len(c.turns)} stop_reason={c.stop_reason} "
              f"tokens_in={c.input_tokens} tokens_out={c.output_tokens}")


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[], ClientProtocol] = client_mod.make_client,
) -> int:
    cfg = _load_config(args.config)
    ws = _resolve_workspace(args.repo, cfg=cfg)

    # SPEC §5.1 candidate flow: when --test is omitted, run discovery +
    # eligibility and pick (or print) candidates.
    target_nodeid = args.test
    if target_nodeid is None or args.list_candidates:
        candidates = _scan_candidates(ws, cfg)
        if args.list_candidates:
            _print_candidates(candidates)
            return 0
        target_nodeid = _auto_select(candidates)
        if target_nodeid is None:
            return 1

    try:
        client = client_factory()
    except client_mod.MissingAPIKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    result = extract_mod.extract(
        ws=ws,
        nodeid=target_nodeid,
        config=cfg,
        client=client,
    )
    _report(result)
    return 0 if result.success else 1
