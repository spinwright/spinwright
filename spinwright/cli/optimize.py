from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from spinwright import config as cfg_mod
from spinwright.llm import client as client_mod
from spinwright.llm.client import ClientProtocol
from spinwright.optimization import optimize as opt_mod
from spinwright.repo import workspace as workspace_mod


def _load_config(path: str | None) -> cfg_mod.Config:
    if path:
        return cfg_mod.load(path)
    return cfg_mod.default()


def _resolve_workspace(workspace_arg: str) -> workspace_mod.Workspace:
    root = Path(workspace_arg).expanduser().resolve()
    return workspace_mod.reuse(root)


def _resolve_extraction(workspace: workspace_mod.Workspace, extraction_arg: str) -> Path:
    p = Path(extraction_arg)
    if not p.is_absolute():
        for candidate in (Path.cwd() / p, workspace.repo_dir / p):
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved
    p = p.resolve()
    if not p.exists():
        raise SystemExit(f"extraction not found: {extraction_arg!r}")
    return p


def _report(result: opt_mod.OptimizationResult) -> None:
    bw = result.baseline_walltime
    cw = result.candidate_walltime
    bcg = result.baseline_callgrind
    ccg = result.candidate_callgrind
    print()
    if result.accepted:
        print(f"Optimization ACCEPTED for {Path(result.nodeid_or_extraction).name}")
    else:
        print(f"Optimization REJECTED for {Path(result.nodeid_or_extraction).name}")
        print(f"  reason: {result.rejection_reason}")

    print(f"  gate metric: {result.gate_metric}")
    if bw is not None:
        print(f"  baseline walltime:  best={bw.best_seconds*1e6:.2f} us  median={bw.median_seconds*1e6:.2f} us  stddev={bw.stddev_seconds*1e6:.2f} us")
    if cw is not None:
        print(f"  candidate walltime: best={cw.best_seconds*1e6:.2f} us  median={cw.median_seconds*1e6:.2f} us  stddev={cw.stddev_seconds*1e6:.2f} us")
    if result.relative_walltime_improvement is not None:
        print(f"  walltime delta:     {result.relative_walltime_improvement:+.2%}")
    if bcg is not None:
        print(f"  baseline callgrind:  {bcg.instructions:,} inst/call  (N={bcg.autoscale_iterations:,})")
    if ccg is not None:
        print(f"  candidate callgrind: {ccg.instructions:,} inst/call  (N={ccg.autoscale_iterations:,})")
    if result.relative_callgrind_improvement is not None:
        print(f"  callgrind delta:    {result.relative_callgrind_improvement:+.2%}")
    if result.relative_improvement is not None:
        print(f"  primary delta:      {result.relative_improvement:+.2%} (threshold {result.threshold:.0%})")
    if result.commit_sha:
        print(f"  commit:    {result.commit_sha}")
    if result.reverted_paths:
        print(f"  reverted:  {len(result.reverted_paths)} file(s) restored to HEAD")
    if result.conversation:
        c = result.conversation
        print(f"  tokens:    in={c.input_tokens} out={c.output_tokens} "
              f"cache_w={c.cache_creation_input_tokens} cache_r={c.cache_read_input_tokens}")
        print(f"  turns:     {len(c.turns)}  stop_reason={c.stop_reason}")
    if result.diff and not result.accepted:
        snippet = "\n".join(result.diff.splitlines()[:40])
        print("  attempted diff (first 40 lines):")
        for line in snippet.splitlines():
            print(f"    {line}")


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[], ClientProtocol] = client_mod.make_client,
) -> int:
    cfg = _load_config(args.config)
    ws = _resolve_workspace(args.workspace)
    extraction = _resolve_extraction(ws, args.extraction)

    try:
        client = client_factory()
    except client_mod.MissingAPIKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    result = opt_mod.optimize_once(
        ws=ws,
        extraction_path=extraction,
        config=cfg,
        client=client,
        extra_excludes=tuple(args.exclude_path),
    )
    _report(result)
    return 0 if result.accepted else 1
