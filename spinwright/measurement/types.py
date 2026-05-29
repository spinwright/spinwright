from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalltimeResult:
    best_seconds: float
    median_seconds: float
    stddev_seconds: float
    iterations_per_repeat: int
    repeats: int


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    error: str | None = None


@dataclass(frozen=True)
class CallgrindResult:
    instructions: int                # per-call (already divided by autoscale N)
    autoscale_iterations: int
    total_inst_at_n_plus_one: int   # raw inst count from the N+1 run
    baseline_inst_at_one: int       # raw inst count from the 1-run baseline
    output_path: str                # path to the N+1 callgrind.out file (for reference)
