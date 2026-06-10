from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from spinwright.repo.workspace import Workspace


@dataclass
class RegressionResult:
    passed: bool  # Final suite state after any reverts.
    dropped_commits: list[str] = field(default_factory=list)
    final_pytest_output: str = ""
    fallback_used: str = "none"  # "none" | "linear_revert" | "drop_all"


def run_regression_check(
    ws: Workspace,
    accepted_commits: list[str],
    venv_python: Path,
    *,
    pytest_args: tuple[str, ...] = (),
    timeout_seconds: float = 1800.0,
) -> RegressionResult:
    """Run the target's full pytest suite. On failure, apply linear-revert
    (Mod 9): try removing one accepted patch at a time; the first removal that
    restores green is permanently dropped, and we recurse on the remaining
    commits. If no single revert works after one pass, fall back to dropping
    all accepted commits (the workspace ends back at ``ws.base_sha``).
    """
    passed, output = _run_pytest(venv_python, ws.repo_dir, pytest_args, timeout_seconds)
    if passed:
        return RegressionResult(passed=True, final_pytest_output=output)

    remaining = list(accepted_commits)
    dropped: list[str] = []
    used_fallback = "none"

    # Linear-revert: try removing each remaining commit one at a time. After a
    # successful drop the workspace is already at the green ``trial`` state, so
    # we stop. If multiple commits would each individually restore green, we
    # drop just the first one (and leave any further investigation to a future
    # bisect-style refinement).
    while remaining:
        culprit_index = -1
        for i in range(len(remaining)):
            trial = remaining[:i] + remaining[i + 1 :]
            _apply_subset(ws, trial)
            ok, output = _run_pytest(
                venv_python, ws.repo_dir, pytest_args, timeout_seconds
            )
            if ok:
                culprit_index = i
                break
        if culprit_index < 0:
            # No single removal restores green — multi-patch interaction.
            # Conservatively drop everything remaining; workspace ends at base.
            _apply_subset(ws, [])
            dropped.extend(remaining)
            remaining = []
            used_fallback = "drop_all"
            passed, output = _run_pytest(
                venv_python, ws.repo_dir, pytest_args, timeout_seconds
            )
            break
        # Successfully restored green by dropping one. Workspace is already
        # in the trial state.
        dropped.append(remaining[culprit_index])
        remaining = remaining[:culprit_index] + remaining[culprit_index + 1 :]
        used_fallback = "linear_revert"
        passed = True
        break

    return RegressionResult(
        passed=passed,
        dropped_commits=dropped,
        final_pytest_output=output,
        fallback_used=used_fallback,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_pytest(
    venv_python: Path,
    repo_dir: Path,
    pytest_args: tuple[str, ...],
    timeout: float,
) -> tuple[bool, str]:
    """Run pytest in ``repo_dir`` via the target venv. Returns (passed, output)."""
    cmd = [
        str(venv_python),
        "-m",
        "pytest",
        "-p",
        "no:randomly",
        "--tb=short",
        "-q",
        *pytest_args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return (
            False,
            f"pytest timed out after {timeout}s\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}",
        )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _git(repo_dir: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return proc.stdout


def _apply_subset(ws: Workspace, commits_to_keep: list[str]) -> None:
    """Reset the working branch to ``base_sha`` and cherry-pick ``commits_to_keep``
    in order. The branch ends with only those commits on top of base."""
    # Abort any in-flight cherry-pick / merge that a prior trial might have left
    # behind before resetting. ``--quit`` is a no-op if no operation is active.
    _git(ws.repo_dir, "cherry-pick", "--quit", check=False)
    _git(ws.repo_dir, "reset", "--hard", ws.base_sha)
    for sha in commits_to_keep:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(ws.repo_dir),
                "-c",
                "user.email=spinwright@localhost",
                "-c",
                "user.name=spinwright",
                "cherry-pick",
                sha,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Conflict — abort and skip this commit (treat as already dropped).
            _git(ws.repo_dir, "cherry-pick", "--abort", check=False)
