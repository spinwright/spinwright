"""Run inside the target venv. Profiles N iterations of run() under cProfile
and emits a JSON summary.

Usage:
    python cprofile_driver.py <extraction_path> <iterations> <include_prefix> [<exclude1> ...]

``include_prefix`` is a positive filter: entries whose source filename does
NOT start with this string are dropped. Pass an empty string to disable.
``exclude`` args are negative substring filters applied to whatever survives
the include filter. The combination lets callers say "show only files under
my repo, but drop the test harness within it."
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
    include_prefix = sys.argv[3]  # empty string = no include filter
    excludes = sys.argv[4:]

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
        fn = filename or ""
        # Positive include filter (when set): drop everything outside the prefix.
        if include_prefix and not fn.startswith(include_prefix):
            continue
        # Negative excludes: drop within the include set.
        if excludes and any(ex in fn for ex in excludes):
            continue
        entries.append(
            {
                "filename": filename,
                "lineno": lineno,
                "funcname": funcname,
                "calls": nc,
                "primitive_calls": cc,
                "tottime": tt,  # time in this function alone
                "cumtime": ct,  # time in this function + all subcalls
                "tottime_per_call": (tt / nc) if nc else 0.0,
                "cumtime_per_call": (ct / nc) if nc else 0.0,
            }
        )

    total = stats.total_tt or 0.0
    sys.stdout.write(
        json.dumps(
            {
                "iterations": iterations,
                "total_seconds": total,
                "entries": entries,
                "verify_passed": verify_passed,
                "verify_error": verify_error,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
