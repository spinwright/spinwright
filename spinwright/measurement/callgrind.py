from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from spinwright import platform as platform_mod
from spinwright.measurement import drivers, walltime
from spinwright.measurement.auto_scale import autoscale_iterations
from spinwright.measurement.runner import DriverError
from spinwright.measurement.types import CallgrindResult, VerifyResult


_DRIVER = Path(drivers.__file__).parent / "callgrind_driver.py"

_SUMMARY_RE = re.compile(r"^summary:\s*(\d+)", re.MULTILINE)


class CallgrindUnavailable(RuntimeError):
    pass


def parse_summary(output_path: Path) -> int:
    """Read the ``summary:`` line out of a callgrind.out.* file.

    Callgrind writes one ``summary: <Ir>`` line per output file, where ``Ir``
    is the total Intel instruction count. If there are multiple events
    columns, the first (which Spinwright forces to ``Ir`` via the
    ``--collect-jumps`` defaults) is the one we want.
    """
    text = output_path.read_text()
    matches = _SUMMARY_RE.findall(text)
    if not matches:
        raise ValueError(f"no 'summary:' line in {output_path}")
    return int(matches[-1])


def measure_callgrind(
    venv_python: Path,
    extraction_path: Path,
    *,
    valgrind_path: str = "valgrind",
    autoscale_min_instructions: int = 1_000_000_000,
    cwd: Path | None = None,
    timeout_seconds: float | None = 1800.0,
) -> tuple[CallgrindResult, VerifyResult]:
    """Two-run subtraction:
        A: N+1 runs → SET + (N+1)·RUN + VER
        B:   1 run  → SET +     1·RUN + VER
        per-call RUN = (A − B) / N
    Both runs include verify() so VER cancels regardless of state shape.
    """
    if not platform_mod.is_linux():
        raise CallgrindUnavailable(
            "Callgrind requires Linux; current platform is unsupported "
            "(macOS has no working Valgrind port)."
        )
    if not _valgrind_present(valgrind_path):
        raise CallgrindUnavailable(
            f"valgrind binary {valgrind_path!r} not found on PATH — "
            "install valgrind or set measurement.callgrind_path"
        )

    effective_cwd = cwd or extraction_path.parent

    # Probe wallclock to estimate per-call cost for auto-scaling.
    probe_wt, probe_vr = walltime.measure(
        venv_python, extraction_path, repeats=1, cwd=effective_cwd,
    )
    if not probe_vr.passed:
        return _empty_callgrind(), probe_vr
    n = autoscale_iterations(probe_wt.best_seconds, autoscale_min_instructions)

    tmpdir = Path(tempfile.mkdtemp(prefix="sw-callgrind-"))
    out_a = tmpdir / "out.a"
    out_b = tmpdir / "out.b"

    inst_a, vr_a = _run_under_callgrind(
        venv_python, extraction_path, n + 1,
        valgrind_path=valgrind_path,
        out_path=out_a,
        cwd=effective_cwd,
        timeout_seconds=timeout_seconds,
    )
    if not vr_a.passed:
        return _empty_callgrind(), vr_a

    inst_b, vr_b = _run_under_callgrind(
        venv_python, extraction_path, 1,
        valgrind_path=valgrind_path,
        out_path=out_b,
        cwd=effective_cwd,
        timeout_seconds=timeout_seconds,
    )
    if not vr_b.passed:
        return _empty_callgrind(), vr_b

    diff = inst_a - inst_b
    per_call = max(diff // n, 0)
    return CallgrindResult(
        instructions=per_call,
        autoscale_iterations=n,
        total_inst_at_n_plus_one=inst_a,
        baseline_inst_at_one=inst_b,
        output_path=str(out_a),
    ), vr_a  # vr_a == vr_b in practice; pick A


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _valgrind_present(valgrind_path: str) -> bool:
    try:
        proc = subprocess.run(
            [valgrind_path, "--version"], capture_output=True, text=True
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0


def _run_under_callgrind(
    venv_python: Path,
    extraction_path: Path,
    iterations: int,
    *,
    valgrind_path: str,
    out_path: Path,
    cwd: Path,
    timeout_seconds: float | None,
) -> tuple[int, VerifyResult]:
    cmd = [
        valgrind_path,
        "--tool=callgrind",
        "--instr-atstart=yes",
        f"--callgrind-out-file={out_path}",
        "--quiet",
        str(venv_python),
        str(_DRIVER),
        str(extraction_path),
        str(iterations),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0 and not out_path.exists():
        raise DriverError(
            f"valgrind exited {proc.returncode} and produced no output file",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    # Parse driver stdout for verify result. Valgrind interleaves nothing of
    # its own when --quiet is set; the entire stdout should be the driver JSON.
    try:
        payload = json.loads(proc.stdout)
        vr = VerifyResult(passed=payload["verify_passed"], error=payload.get("verify_error"))
    except (json.JSONDecodeError, KeyError):
        vr = VerifyResult(passed=False, error=f"driver stdout was not JSON: {proc.stdout!r}")

    inst = parse_summary(out_path)
    return inst, vr


def _empty_callgrind() -> CallgrindResult:
    return CallgrindResult(
        instructions=0,
        autoscale_iterations=0,
        total_inst_at_n_plus_one=0,
        baseline_inst_at_one=0,
        output_path="",
    )
