from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from spinwright import config as cfg_mod
from spinwright import platform as platform_mod
from spinwright.measurement import callgrind as callgrind_mod
from spinwright.measurement import walltime
from spinwright.measurement.runner import DriverError


def _load_config(path: str | None) -> cfg_mod.Config:
    if path:
        return cfg_mod.load(path)
    return cfg_mod.default()


def _resolve_workspace(workspace_arg: str) -> tuple[Path, Path]:
    """Returns (workspace_root, venv_python). Errors out if the workspace
    doesn't look like one built by `spinwright prep`."""
    root = Path(workspace_arg).expanduser().resolve()
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(
            f"workspace {root} has no .venv/bin/python — run `spinwright prep` first"
        )
    return root, venv_python


def _resolve_extraction(workspace_root: Path, extraction_arg: str) -> Path:
    p = Path(extraction_arg)
    if not p.is_absolute():
        # Try relative to cwd first, then relative to the workspace's repo dir.
        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        repo_candidate = (workspace_root / "repo" / p).resolve()
        if repo_candidate.exists():
            return repo_candidate
    p = p.resolve()
    if not p.exists():
        raise SystemExit(f"extraction not found: {extraction_arg!r}")
    return p


def _print_human(wt, vr, *, callgrind=None, callgrind_skip_reason=None) -> None:
    print("Wallclock:")
    print(f"  best:    {wt.best_seconds * 1e6:12.3f} us")
    print(f"  median:  {wt.median_seconds * 1e6:12.3f} us")
    print(f"  stddev:  {wt.stddev_seconds * 1e6:12.3f} us")
    print(f"  iters per repeat: {wt.iterations_per_repeat}")
    print(f"  repeats: {wt.repeats}")
    if callgrind is not None:
        print("Callgrind:")
        print(f"  per-call instructions: {callgrind.instructions:,}")
        print(f"  autoscale iterations:  {callgrind.autoscale_iterations:,}")
        print(f"  raw A (N+1 runs):      {callgrind.total_inst_at_n_plus_one:,}")
        print(f"  raw B (1 run):         {callgrind.baseline_inst_at_one:,}")
        print(f"  output file:           {callgrind.output_path}")
    elif callgrind_skip_reason:
        print(f"Callgrind: skipped ({callgrind_skip_reason})")
    print(f"Verify: {'PASSED' if vr.passed else 'FAILED'}")
    if vr.error:
        print("Verify error:")
        print(vr.error)


def run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    workspace_root, venv_python = _resolve_workspace(args.workspace)
    extraction = _resolve_extraction(workspace_root, args.extraction)
    repeats = args.repeats if args.repeats is not None else cfg.measurement.walltime_repeats

    try:
        wt, vr = walltime.measure(
            venv_python, extraction, repeats=repeats, cwd=workspace_root / "repo",
        )
    except DriverError as e:
        print(f"measurement driver failed (rc={e.returncode}):", file=sys.stderr)
        print(e.stderr or e.stdout, file=sys.stderr)
        return 2

    cg = None
    cg_skip_reason = None
    if not platform_mod.is_linux():
        cg_skip_reason = "macOS — Linux-only metric"
    else:
        try:
            cg, cg_vr = callgrind_mod.measure_callgrind(
                venv_python, extraction,
                valgrind_path=cfg.measurement.callgrind_path,
                autoscale_min_instructions=cfg.measurement.autoscale_min_instructions,
                cwd=workspace_root / "repo",
            )
            # Prefer the callgrind verify result if it disagrees (unlikely; flagged).
            if not cg_vr.passed and vr.passed:
                vr = cg_vr
        except callgrind_mod.CallgrindUnavailable as e:
            cg_skip_reason = str(e)

    if args.json:
        payload = {
            "walltime": asdict(wt),
            "verify": asdict(vr),
            "callgrind": asdict(cg) if cg is not None else None,
            "callgrind_skip_reason": cg_skip_reason,
            "extraction": str(extraction),
            "workspace": str(workspace_root),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        _print_human(wt, vr, callgrind=cg, callgrind_skip_reason=cg_skip_reason)
        if vr.passed:
            # Hint at the next step so the chain stays paste-able.
            try:
                rel = str(extraction.resolve().relative_to((workspace_root / "repo").resolve()))
            except ValueError:
                rel = str(extraction)
            print()
            print(f"  next: spinwright optimize {workspace_root} --extraction {rel}")
            print(f"   or:  spinwright run      {workspace_root} --extraction {rel}")

    return 0 if vr.passed else 1
