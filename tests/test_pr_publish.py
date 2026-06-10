from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from spinwright.config import PRConfig
from spinwright.pr import publish
from spinwright.pr.builder import PRDraft
from spinwright.repo.workspace import Workspace


def _ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    repo = root / "repo"
    repo.mkdir()
    return Workspace(
        root=root,
        repo_dir=repo,
        venv_dir=root / ".venv",
        branch="spinwright/test",
        base_sha="abc123",
        keep=True,
    )


def _draft() -> PRDraft:
    return PRDraft(title="perf(x): test", body="b", accepted_count=1, dropped_count=0)


def test_local_mode_is_noop(tmp_path: Path):
    result = publish.publish(
        ws=_ws(tmp_path),
        pr_draft=_draft(),
        pr_config=PRConfig(
            mode="local", base_branch="main", branch_prefix="spinwright/"
        ),
        run_dir=tmp_path / "run",
    )
    assert result.mode == "local"
    assert result.skipped_reason is None
    assert result.pr_url is None


def test_github_action_mode_falls_back_when_gh_missing(tmp_path: Path):
    with patch("spinwright.pr.publish.shutil.which", return_value=None):
        result = publish.publish(
            ws=_ws(tmp_path),
            pr_draft=_draft(),
            pr_config=PRConfig(
                mode="github_action", base_branch="main", branch_prefix="spinwright/"
            ),
            run_dir=tmp_path / "run",
        )
    assert result.mode == "local"
    assert "gh CLI not found" in (result.skipped_reason or "")


def test_github_action_pushes_and_creates_pr(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "PR.md").write_text("# PR\n\nbody\n")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="https://github.com/org/repo/pull/42\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="?")

    with (
        patch("spinwright.pr.publish.shutil.which", return_value="/usr/bin/gh"),
        patch("spinwright.pr.publish.subprocess.run", side_effect=fake_run),
    ):
        result = publish.publish(
            ws=_ws(tmp_path),
            pr_draft=_draft(),
            pr_config=PRConfig(
                mode="github_action", base_branch="main", branch_prefix="spinwright/"
            ),
            run_dir=run_dir,
        )
    assert result.mode == "github_action"
    assert result.pr_url == "https://github.com/org/repo/pull/42"
    assert result.pushed_branch == "spinwright/test"
    # Verify both commands were invoked
    assert any(c[0] == "git" and "push" in c for c in calls)
    assert any(c[0] == "gh" and "pr" in c and "create" in c for c in calls)


def test_github_action_handles_push_failure(tmp_path: Path):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="fatal: remote rejected\n",
        )

    with (
        patch("spinwright.pr.publish.shutil.which", return_value="/usr/bin/gh"),
        patch("spinwright.pr.publish.subprocess.run", side_effect=fake_run),
    ):
        result = publish.publish(
            ws=_ws(tmp_path),
            pr_draft=_draft(),
            pr_config=PRConfig(
                mode="github_action", base_branch="main", branch_prefix="spinwright/"
            ),
            run_dir=tmp_path / "run",
        )
    assert result.mode == "local"
    assert "git push failed" in (result.skipped_reason or "")


def test_github_action_handles_gh_failure(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "PR.md").write_text("# PR\n\nbody\n")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="authentication failed\n",
            )

    with (
        patch("spinwright.pr.publish.shutil.which", return_value="/usr/bin/gh"),
        patch("spinwright.pr.publish.subprocess.run", side_effect=fake_run),
    ):
        result = publish.publish(
            ws=_ws(tmp_path),
            pr_draft=_draft(),
            pr_config=PRConfig(
                mode="github_action", base_branch="main", branch_prefix="spinwright/"
            ),
            run_dir=run_dir,
        )
    assert result.mode == "github_action"
    assert result.pr_url is None
    assert result.pushed_branch == "spinwright/test"  # push did succeed
    assert "gh pr create failed" in (result.skipped_reason or "")
