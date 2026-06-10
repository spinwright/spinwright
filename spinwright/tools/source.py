from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from spinwright.tools import drivers


_READ_SOURCE_DRIVER = Path(drivers.__file__).parent / "read_source_driver.py"


def list_tests(
    venv_python: Path, repo_dir: Path, *, pattern: str | None = None
) -> list[str]:
    """Enumerate pytest nodeids in ``repo_dir`` via ``pytest --collect-only -q``.

    Cheap: no test bodies are executed. If ``pattern`` is given, it's passed
    as ``-k`` to pytest.
    """
    cmd = [str(venv_python), "-m", "pytest", "--collect-only", "-q"]
    if pattern:
        cmd.extend(["-k", pattern])
    proc = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True)
    nodeids: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("="):
            nodeids.append(line)
    return nodeids


def get_test_source(repo_dir: Path, nodeid: str) -> dict:
    """Return source slice + metadata for the test identified by ``nodeid``.

    Result shape: {path, source, lineno, end_lineno, decorators, kind}.
    """
    rel_path, *parts = nodeid.split("::")
    if not parts:
        raise ValueError(f"nodeid has no test component: {nodeid!r}")
    parts[-1] = parts[-1].split("[", 1)[0]  # strip parametrize id

    abs_path = (repo_dir / rel_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"test file not found: {abs_path}")

    source_text = abs_path.read_text()
    tree = ast.parse(source_text, filename=str(abs_path))

    func, class_node = _find_test_node(tree, parts)
    if func is None:
        raise LookupError(f"test {nodeid!r} not found in {abs_path}")

    func_source = ast.get_source_segment(source_text, func) or ""
    decorators = [
        f"@{ast.get_source_segment(source_text, d) or ''}" for d in func.decorator_list
    ]
    return {
        "path": str(abs_path),
        "nodeid": nodeid,
        "kind": "method" if class_node is not None else "function",
        "class_name": class_node.name if class_node is not None else None,
        "lineno": func.lineno,
        "end_lineno": func.end_lineno,
        "decorators": decorators,
        "source": func_source,
    }


def read_source(venv_python: Path, qualname: str) -> dict:
    """Resolve a Python dotted qualname to its source code, via a subprocess
    in the target venv (so the importable namespace matches the test runs)."""
    proc = subprocess.run(
        [str(venv_python), str(_READ_SOURCE_DRIVER), qualname],
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(proc.stdout) if proc.stdout else {}
    except json.JSONDecodeError:
        payload = {"error": f"driver produced non-JSON: {proc.stdout!r}"}
    if "error" in payload:
        raise LookupError(payload["error"])
    return payload


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_test_node(
    tree: ast.Module, parts: list[str]
) -> tuple[ast.FunctionDef | None, ast.ClassDef | None]:
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == parts[0]:
                return node, None
        return None, None
    if len(parts) == 2:
        class_name, method_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return item, node
    return None, None
