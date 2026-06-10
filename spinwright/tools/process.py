from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def run_python(
    venv_python: Path,
    code: str,
    *,
    cwd: Path,
    timeout_seconds: float = 30.0,
) -> RunResult:
    """Execute ``code`` via the target venv's Python in a fresh subprocess.

    ``cwd`` should be inside the workspace. Stdout/stderr are captured; the
    process is killed on timeout (``timed_out=True`` in the result). Used by
    the LLM to sanity-check imports, import a fresh extraction, etc.
    """
    try:
        proc = subprocess.run(
            [str(venv_python), "-c", code],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(
            returncode=-1,
            stdout=e.stdout.decode()
            if isinstance(e.stdout, bytes)
            else (e.stdout or ""),
            stderr=e.stderr.decode()
            if isinstance(e.stderr, bytes)
            else (e.stderr or ""),
            timed_out=True,
        )
    return RunResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        timed_out=False,
    )
