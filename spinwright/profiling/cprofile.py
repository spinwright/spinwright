from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spinwright.measurement.runner import run_driver
from spinwright.profiling import drivers


_DRIVER = Path(drivers.__file__).parent / "cprofile_driver.py"


@dataclass(frozen=True)
class ProfileEntry:
    funcname: str
    filename: str
    lineno: int
    calls: int
    primitive_calls: int
    tottime: float
    cumtime: float
    tottime_per_call: float
    cumtime_per_call: float


@dataclass(frozen=True)
class CProfileResult:
    iterations: int
    total_seconds: float
    entries: tuple[ProfileEntry, ...]
    verify_passed: bool
    verify_error: str | None


def profile_cprofile(
    venv_python: Path,
    extraction_path: Path,
    *,
    iterations: int = 1000,
    include_prefix: str | None = None,
    exclude_paths: tuple[str, ...] = (),
    cwd: Path | None = None,
    timeout_seconds: float | None = 120.0,
) -> CProfileResult:
    """Profile ``iterations`` calls of the extraction's ``run`` under cProfile.

    ``include_prefix`` (when set) is a positive filter: entries whose source
    filename does NOT start with this string are dropped before the result is
    returned. Pass the target-repo root to limit output to user code; pass
    ``None`` to see everything (interpreter internals included).

    ``exclude_paths`` are substring-matched against each entry's source filename
    AFTER the include filter; matching entries are dropped. Use this to silence
    in-included-set noise (e.g. an extraction harness directory) — for the
    common "show only my repo" case, set ``include_prefix`` instead.
    """
    effective_cwd = cwd or extraction_path.parent
    payload = run_driver(
        venv_python,
        _DRIVER,
        [str(extraction_path), str(iterations), include_prefix or "", *exclude_paths],
        cwd=effective_cwd,
        timeout_seconds=timeout_seconds,
    )
    entries = tuple(
        ProfileEntry(
            funcname=e["funcname"],
            filename=e["filename"] or "",
            lineno=e["lineno"],
            calls=e["calls"],
            primitive_calls=e["primitive_calls"],
            tottime=e["tottime"],
            cumtime=e["cumtime"],
            tottime_per_call=e["tottime_per_call"],
            cumtime_per_call=e["cumtime_per_call"],
        )
        for e in payload["entries"]
    )
    return CProfileResult(
        iterations=payload["iterations"],
        total_seconds=payload["total_seconds"],
        entries=entries,
        verify_passed=payload["verify_passed"],
        verify_error=payload["verify_error"],
    )


def top_entries(
    result: CProfileResult,
    *,
    by: str = "cumtime",
    limit: int = 20,
) -> tuple[ProfileEntry, ...]:
    """Sort entries by 'tottime' or 'cumtime' and slice to ``limit``."""
    if by not in {"tottime", "cumtime", "tottime_per_call", "cumtime_per_call"}:
        raise ValueError(f"unknown sort key: {by!r}")
    return tuple(
        sorted(result.entries, key=lambda e: getattr(e, by), reverse=True)[:limit]
    )
