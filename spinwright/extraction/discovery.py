from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodeDuration:
    nodeid: str
    seconds: float
    phase: str  # "call" | "setup" | "teardown"


@dataclass(frozen=True)
class DiscoveryReport:
    """Returned by ``discover_verbose`` so callers can show diagnostics when
    no slow tests come back. ``returncode`` 5 means "no tests collected"
    (pytest's exit code for that case)."""

    durations: list[NodeDuration]
    returncode: int
    stdout_tail: str
    stderr_tail: str


_DURATION_LINE_RE = re.compile(
    r"^\s*(?P<seconds>\d+(?:\.\d+)?)s\s+(?P<phase>call|setup|teardown)\s+(?P<nodeid>\S.*\S)\s*$"
)
_DURATION_HEADER_RE = re.compile(r"slowest\s*\S*\s*durations", re.IGNORECASE)


def _parse_durations(stdout: str) -> list[NodeDuration]:
    durations: list[NodeDuration] = []
    in_section = False
    for line in stdout.splitlines():
        if not in_section:
            if _DURATION_HEADER_RE.search(line):
                in_section = True
            continue
        m = _DURATION_LINE_RE.match(line)
        if m:
            durations.append(
                NodeDuration(
                    nodeid=m.group("nodeid"),
                    seconds=float(m.group("seconds")),
                    phase=m.group("phase"),
                )
            )
        elif line.startswith("=") and durations:
            break
    return durations


def discover(
    venv_python: Path,
    repo_dir: Path,
    *,
    slow_threshold_seconds: float = 0.1,
    pytest_paths: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
    timeout_seconds: float | None = None,
) -> list[NodeDuration]:
    """Run pytest with --durations=0 and return call durations sorted descending.

    Only the "call" phase is returned (setup/teardown excluded). Tests faster
    than ``slow_threshold_seconds`` are filtered out. Failing tests are
    tolerated (we still parse whatever durations were emitted), but errors
    that prevent collection are silently absorbed — use ``discover_verbose``
    when you need to surface those.
    """
    return discover_verbose(
        venv_python, repo_dir,
        slow_threshold_seconds=slow_threshold_seconds,
        pytest_paths=pytest_paths, extra_args=extra_args,
        timeout_seconds=timeout_seconds,
    ).durations


def discover_verbose(
    venv_python: Path,
    repo_dir: Path,
    *,
    slow_threshold_seconds: float = 0.1,
    pytest_paths: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
    timeout_seconds: float | None = None,
) -> DiscoveryReport:
    """Same as ``discover`` but also returns pytest's return code and stdout/
    stderr tails so the caller can surface collection failures."""
    cmd = [
        str(venv_python),
        "-m", "pytest",
        "--durations=0",
        "-p", "no:randomly",
        "--tb=no",
        "-q",
        *extra_args,
        *pytest_paths,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    durations = _parse_durations(proc.stdout)
    calls = [d for d in durations if d.phase == "call" and d.seconds >= slow_threshold_seconds]
    calls.sort(key=lambda d: d.seconds, reverse=True)
    return DiscoveryReport(
        durations=calls,
        returncode=proc.returncode,
        stdout_tail="\n".join(proc.stdout.splitlines()[-25:]),
        stderr_tail="\n".join(proc.stderr.splitlines()[-25:]),
    )
