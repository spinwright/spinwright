"""Run inside the target venv. Measures an extraction module's run() under
timeit and emits a single JSON line on stdout.

Usage:
    python walltime_driver.py <extraction_path> <repeats>
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import timeit
import traceback


# Allow the extraction to import modules from the working directory
# (e.g., `from target_pkg import ...` when the target package lives at
# <cwd>/target_pkg). Without this, `python script.py` mode adds only the
# driver's own directory to sys.path.
sys.path.insert(0, os.getcwd())


def _load(path: str):
    spec = importlib.util.spec_from_file_location("_sw_extraction", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    extraction_path = sys.argv[1]
    repeats = int(sys.argv[2])

    mod = _load(extraction_path)
    state = mod.setup()

    timer = timeit.Timer(lambda: mod.run(state))
    iterations_per_repeat, _ = timer.autorange()
    totals = timer.repeat(repeat=repeats, number=iterations_per_repeat)
    per_call = [t / iterations_per_repeat for t in totals]

    verify_passed = True
    verify_error: str | None = None
    try:
        mod.verify(state)
    except Exception:
        verify_passed = False
        verify_error = traceback.format_exc()

    result = {
        "best_seconds": min(per_call),
        "median_seconds": statistics.median(per_call),
        "stddev_seconds": statistics.stdev(per_call) if len(per_call) > 1 else 0.0,
        "iterations_per_repeat": iterations_per_repeat,
        "repeats": len(per_call),
        "verify_passed": verify_passed,
        "verify_error": verify_error,
    }
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
