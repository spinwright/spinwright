from __future__ import annotations


def autoscale_iterations(
    per_call_seconds: float,
    min_total_instructions: int,
    *,
    instructions_per_second_estimate: float = 1.0e9,
    cap: int = 10_000_000,
) -> int:
    """Pick an iteration count N so total instrumented work crosses a threshold.

    We don't know the exact instructions-per-call without running Callgrind
    first; the heuristic uses a wallclock probe and the rough rule that a
    modern x86 CPU dispatches on the order of 1e9 instructions/second of
    user-mode work (close enough for choosing batch sizes — the precise count
    comes from Callgrind itself).

    Returns at least 1 and at most ``cap`` (a sanity ceiling so a probe that
    measures noise as ~0 doesn't ask for a 10-billion-iteration loop).
    """
    if per_call_seconds <= 0:
        return 1
    estimated_per_call = max(
        int(per_call_seconds * instructions_per_second_estimate), 1
    )
    n = (min_total_instructions + estimated_per_call - 1) // estimated_per_call
    return max(1, min(n, cap))
