from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import spinwright
from spinwright.optimization.loop import LoopResult
from spinwright.optimization.regression import RegressionResult
from spinwright.pr.builder import PRDraft


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_run_id(now: dt.datetime | None = None) -> str:
    """``run_20260529T120000_a1b2c3`` — timestamp prefix so directories sort
    chronologically, plus a short random suffix to avoid collisions on
    quick-fire runs."""
    when = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%S")
    # Use the microsecond component as the entropy source — it's cheap, doesn't
    # require importing secrets, and is unique enough at second-resolution.
    micro = (now or dt.datetime.now(dt.timezone.utc)).strftime("%f")
    return f"run_{when}_{micro[:6]}"


def write_run_directory(
    *,
    runs_root: Path,
    run_id: str,
    pr_draft: PRDraft | None,
    loop_result: LoopResult,
    regression: RegressionResult | None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Create ``<runs_root>/<run_id>/`` and write the run's artifacts.

    Writes:
      - ``PR.md`` (only when ``pr_draft`` is provided): the rendered PR body,
        with the title as the H1 above it.
      - ``run_summary.json``: serialized LoopResult + RegressionResult + any
        ``extra_metadata`` the caller passes (e.g. config, workspace path).

    Returns the path to the created run directory.
    """
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if pr_draft is not None:
        pr_path = run_dir / "PR.md"
        pr_path.write_text(f"# {pr_draft.title}\n\n{pr_draft.body}")

    summary = {
        "spinwright_version": spinwright.__version__,
        "run_id": run_id,
        "loop_result": _serialize(loop_result),
        "regression": _serialize(regression),
        "pr": (
            {
                "title": pr_draft.title,
                "accepted_count": pr_draft.accepted_count,
                "dropped_count": pr_draft.dropped_count,
            }
            if pr_draft is not None
            else None
        ),
    }
    if extra_metadata:
        summary["metadata"] = extra_metadata
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    return run_dir


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize(value: Any) -> Any:
    """Recursively convert dataclasses, Paths, and tuples to JSON-friendly
    primitives. ``ConversationResult`` and friends are dataclasses so
    ``asdict`` covers them; the helper here also handles the misc cases that
    slip through (Paths inside dataclasses, in particular)."""
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
