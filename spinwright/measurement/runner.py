from __future__ import annotations

import json
import subprocess
from pathlib import Path


class DriverError(RuntimeError):
    """Raised when a measurement driver subprocess fails to produce parseable JSON."""

    def __init__(
        self, message: str, *, returncode: int, stdout: str, stderr: str
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_driver(
    venv_python: Path,
    driver_path: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    """Invoke ``driver_path`` with ``venv_python`` and parse its stdout as JSON.

    The driver is expected to write a single JSON object to stdout and nothing
    else. Anything on stderr is forwarded into the raised exception if parsing
    fails. A non-zero exit code also raises ``DriverError``.
    """
    cmd = [str(venv_python), str(driver_path), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise DriverError(
            f"driver {driver_path.name!r} exited {proc.returncode}",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise DriverError(
            f"driver {driver_path.name!r} produced non-JSON stdout: {e}",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        ) from e
