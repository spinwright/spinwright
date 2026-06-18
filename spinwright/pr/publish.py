from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spinwright.config import PRConfig
from spinwright.optimization.loop import LoopResult
from spinwright.optimization.optimize import OptimizationResult
from spinwright.pr.builder import PRDraft, diff_rel_paths
from spinwright.repo import workspace as workspace_mod
from spinwright.repo.workspace import Workspace
from spinwright.tools import git as git_mod


@dataclass(frozen=True)
class PublishResult:
    mode: str  # "local" | "github_action"
    pr_url: str | None  # set when github_action successfully opened a PR
    pushed_branch: str | None
    skipped_reason: str | None  # set when the publish step was skipped


def pick_review_attempt(loop_result: LoopResult) -> OptimizationResult | None:
    """Choose the single most-improved attempt that produced a code change, for
    publishing under ``--always-publish`` when nothing cleared the gate.

    Used in the "no surviving patches" case: every rejected attempt was reverted
    (and any accepted-then-regressed commit was stripped), so the branch is back
    at base — but each attempt's diff is preserved on its ``OptimizationResult``.
    Ranking by ``relative_improvement`` surfaces the attempt that came closest to
    passing. Returns None when no iteration made an edit.
    """
    best: OptimizationResult | None = None
    best_key = float("-inf")
    for it in loop_result.iterations:
        if not (it.diff or "").strip():
            continue
        key = (
            it.relative_improvement
            if it.relative_improvement is not None
            else float("-inf")
        )
        if best is None or key > best_key:
            best, best_key = it, key
    return best


def commit_attempt_for_review(ws: Workspace, attempt: OptimizationResult) -> str | None:
    """Re-apply a reverted attempt's diff and commit it on the working branch so
    it can be published for review under ``--always-publish``.

    Assumes the tree is clean at base (true after the loop reverts rejected
    patches and the regression check resets dropped ones). Returns the commit
    SHA, or None if the diff is empty or no longer applies cleanly. Mutates
    ``attempt.commit_sha`` so the PR body can reference the commit.
    """
    diff = attempt.diff or ""
    paths = [ws.repo_dir / p for p in diff_rel_paths(diff)]
    if not paths or not git_mod.git_apply(ws.repo_dir, diff):
        return None
    reason = (attempt.rejection_reason or "below gate threshold").splitlines()[0]
    sha = workspace_mod.commit(
        ws, paths, f"spinwright: unaccepted attempt for review ({reason})"
    )
    attempt.commit_sha = sha
    return sha


def publish(
    *,
    ws: Workspace,
    pr_draft: PRDraft,
    pr_config: PRConfig,
    run_dir: Path,
) -> PublishResult:
    """Local mode is always a no-op (the run directory already contains
    ``PR.md``). GitHub-Action mode pushes the working branch and calls
    ``gh pr create --title --body-file``. If ``gh`` is missing or the push
    fails, we fall back to local-mode with the reason captured."""
    if pr_config.mode != "github_action":
        return PublishResult(
            mode="local", pr_url=None, pushed_branch=None, skipped_reason=None
        )

    if shutil.which("gh") is None:
        return PublishResult(
            mode="local",
            pr_url=None,
            pushed_branch=None,
            skipped_reason="gh CLI not found on PATH; falling back to local PR.md",
        )

    push_proc = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "push", "-u", "origin", ws.branch],
        capture_output=True,
        text=True,
    )
    if push_proc.returncode != 0:
        return PublishResult(
            mode="local",
            pr_url=None,
            pushed_branch=None,
            skipped_reason=f"git push failed: {push_proc.stderr.strip()[:200]}",
        )

    body_path = run_dir / "PR.md"
    pr_proc = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            pr_draft.title,
            "--body-file",
            str(body_path),
            "--base",
            pr_config.base_branch,
            "--head",
            ws.branch,
        ],
        cwd=str(ws.repo_dir),
        capture_output=True,
        text=True,
    )
    if pr_proc.returncode != 0:
        return PublishResult(
            mode="github_action",
            pr_url=None,
            pushed_branch=ws.branch,
            skipped_reason=f"gh pr create failed: {pr_proc.stderr.strip()[:200]}",
        )
    return PublishResult(
        mode="github_action",
        pr_url=pr_proc.stdout.strip() or None,
        pushed_branch=ws.branch,
        skipped_reason=None,
    )
