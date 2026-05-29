from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from spinwright import config as cfg_mod
from spinwright import platform as platform_mod
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


def _print_human(wt, vr, *, callgrind_skipped: bool) -> None:
    print("Wallclock:")
    print(f"  best:    {wt.best_seconds * 1e6:12.3f} us")
    print(f"  median:  {wt.median_seconds * 1e6:12.3f} us")
    print(f"  stddev:  {wt.stddev_seconds * 1e6:12.3f} us")
    print(f"  iters per repeat: {wt.iterations_per_repeat}")
    print(f"  repeats: {wt.repeats}")
    if callgrind_skipped:
        print("Callgrind: skipped (macOS — Linux-only metric)")
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

    callgrind_skipped = platform_mod.is_macos()  # step 6 will gate this differently on Linux

    if args.json:
        payload = {
            "walltime": asdict(wt),
            "verify": asdict(vr),
            "callgrind": None if callgrind_skipped else "not_yet_implemented",
            "extraction": str(extraction),
            "workspace": str(workspace_root),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        _print_human(wt, vr, callgrind_skipped=callgrind_skipped)

    return 0 if vr.passed else 1
