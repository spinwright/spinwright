from __future__ import annotations

from pathlib import Path

from spinwright.measurement import drivers
from spinwright.measurement.runner import run_driver
from spinwright.measurement.types import VerifyResult, WalltimeResult


_WALLTIME_DRIVER = Path(drivers.__file__).parent / "walltime_driver.py"


def measure(
    venv_python: Path,
    extraction_path: Path,
    *,
    repeats: int = 5,
    cwd: Path | None = None,
    timeout_seconds: float | None = None,
) -> tuple[WalltimeResult, VerifyResult]:
    """Run the extraction under timeit in a subprocess of ``venv_python``.

    Returns the wallclock result and a separate verify result. ``verify()`` is
    invoked once inside the driver after the measurement loop completes.
    ``cwd`` defaults to the extraction's parent directory so extractions that
    do ``from target_pkg import ...`` resolve packages co-located in the repo.
    """
    effective_cwd = cwd or extraction_path.parent
    payload = run_driver(
        venv_python,
        _WALLTIME_DRIVER,
        [str(extraction_path), str(repeats)],
        cwd=effective_cwd,
        timeout_seconds=timeout_seconds,
    )
    walltime = WalltimeResult(
        best_seconds=payload["best_seconds"],
        median_seconds=payload["median_seconds"],
        stddev_seconds=payload["stddev_seconds"],
        iterations_per_repeat=payload["iterations_per_repeat"],
        repeats=payload["repeats"],
    )
    verify = VerifyResult(
        passed=payload["verify_passed"],
        error=payload["verify_error"],
    )
    return walltime, verify
