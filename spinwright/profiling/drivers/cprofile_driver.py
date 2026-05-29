"""Run inside the target venv. Profiles N iterations of run() under cProfile
and emits a JSON summary.

Usage:
    python cprofile_driver.py <extraction_path> <iterations> [<exclude_prefix1> ...]

Excludes filter functions whose source filename matches any of the given path
prefixes (substring match). Stdlib and third-party noise can be silenced this
way without losing the ability to see them in a "no excludes" call.
"""

from __future__ import annotations

import cProfile
import importlib.util
import json
import os
import pstats
import sys
import traceback


# Make the working directory importable so `from target_pkg import ...` in
# the extraction resolves the in-repo package (same fix as walltime_driver).
sys.path.insert(0, os.getcwd())


def _load(path):
    spec = importlib.util.spec_from_file_location("_sw_extraction", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    extraction_path = sys.argv[1]
    iterations = int(sys.argv[2])
    excludes = sys.argv[3:]

    mod = _load(extraction_path)
    state = mod.setup()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(iterations):
        mod.run(state)
    profiler.disable()

    verify_passed = True
    verify_error = None
    try:
        mod.verify(state)
    except Exception:
        verify_passed = False
        verify_error = traceback.format_exc()

    stats = pstats.Stats(profiler)
    entries = []
    for (filename, lineno, funcname), (cc, nc, tt, ct, callers) in stats.stats.items():
        # filter out excluded paths
        if excludes and any(ex in (filename or "") for ex in excludes):
            continue
        entries.append({
            "filename": filename,
            "lineno": lineno,
            "funcname": funcname,
            "calls": nc,
            "primitive_calls": cc,
            "tottime": tt,        # time in this function alone
            "cumtime": ct,        # time in this function + all subcalls
            "tottime_per_call": (tt / nc) if nc else 0.0,
            "cumtime_per_call": (ct / nc) if nc else 0.0,
        })

    total = stats.total_tt or 0.0
    sys.stdout.write(json.dumps({
        "iterations": iterations,
        "total_seconds": total,
        "entries": entries,
        "verify_passed": verify_passed,
        "verify_error": verify_error,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
