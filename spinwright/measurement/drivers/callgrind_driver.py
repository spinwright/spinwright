"""Run inside the target venv, under valgrind --tool=callgrind. Runs
setup() + N × run(state) + verify(state) so the only knob that changes
between Spinwright's two passes is N.

Usage:
    python callgrind_driver.py <extraction_path> <iterations>
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback


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
    iterations = int(sys.argv[2])

    mod = _load(extraction_path)
    state = mod.setup()
    for _ in range(iterations):
        mod.run(state)

    verify_passed = True
    verify_error = None
    try:
        mod.verify(state)
    except Exception:
        verify_passed = False
        verify_error = traceback.format_exc()

    sys.stdout.write(
        json.dumps(
            {
                "iterations": iterations,
                "verify_passed": verify_passed,
                "verify_error": verify_error,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
