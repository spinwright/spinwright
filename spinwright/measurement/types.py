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
