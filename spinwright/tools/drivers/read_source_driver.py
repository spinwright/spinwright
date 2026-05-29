"""Run inside the target venv. Resolves a Python dotted qualname to its source
file and source code via inspect, emits JSON.

Usage:
    python read_source_driver.py <qualname>
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import traceback


def resolve(qualname: str):
    parts = qualname.split(".")
    # Try the longest module prefix that's importable; walk the remainder
    # with getattr. This handles both "pkg.mod.Class.method" and
    # "pkg.mod.func" without the caller needing to know where the module
    # ends and the attribute path begins.
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        obj = mod
        try:
            for part in parts[i:]:
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj, mod_name
    raise LookupError(f"could not resolve {qualname!r}")


def main() -> int:
    qualname = sys.argv[1]
    try:
        obj, module = resolve(qualname)
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source, lineno = inspect.getsourcelines(obj)
        payload = {
            "qualname": qualname,
            "module": module,
            "path": source_file,
            "lineno": lineno,
            "source": "".join(source),
        }
    except Exception:
        payload = {"error": traceback.format_exc()}
    sys.stdout.write(json.dumps(payload))
    return 0 if "error" not in payload else 1


if __name__ == "__main__":
    raise SystemExit(main())
