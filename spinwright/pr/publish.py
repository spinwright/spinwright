from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spinwright.config import PRConfig
from spinwright.pr.builder import PRDraft
from spinwright.repo.workspace import Workspace


@dataclass(frozen=True)
class PublishResult:
    mode: str  # "local" | "github_action"
    pr_url: str | None  # set when github_action successfully opened a PR
    pushed_branch: str | None
    skipped_reason: str | None  # set when the publish step was skipped


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
