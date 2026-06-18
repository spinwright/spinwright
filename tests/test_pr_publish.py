from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from spinwright.config import PRConfig
from spinwright.llm.dispatch import ConversationResult
from spinwright.optimization.loop import LoopResult
from spinwright.optimization.optimize import OptimizationResult
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


def _attempt(
    *, diff: str, rel: float | None, reason: str | None = "below threshold"
) -> OptimizationResult:
    return OptimizationResult(
        accepted=False,
        nodeid_or_extraction="x",
        baseline_walltime=None,
        candidate_walltime=None,
        baseline_callgrind=None,
        candidate_callgrind=None,
        candidate_verify=None,
        relative_improvement=rel,
        relative_walltime_improvement=rel,
        relative_callgrind_improvement=None,
        gate_metric="walltime_median",
        threshold=0.20,
        diff=diff,
        commit_sha=None,
        rejection_reason=reason,
        conversation=ConversationResult(
            stop_reason="end_turn", turns=[], final_text="tried it"
        ),
        extraction_path=Path("x.py"),
    )


def _loop(*iterations: OptimizationResult) -> LoopResult:
    return LoopResult(
        success=True,
        extraction_path=Path("x.py"),
        baseline=None,
        iterations=list(iterations),
        accepted_indices=[],
        final_walltime=None,
        final_callgrind=None,
        explored=[],
        stop_reason="max_iterations",
    )


def _git_workspace(tmp_path: Path) -> Workspace:
    """Real git repo on a working branch with one committed file, ready to
    accept a re-applied diff."""
    root = tmp_path / "ws"
    repo = root / "repo"
    repo.mkdir(parents=True)
    (repo / "mod.py").write_text("def x():\n    return 1\n")
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "-c", "user.email=a@a", "-c", "user.name=a", "add", "."],
        ["git", "-c", "user.email=a@a", "-c", "user.name=a", "commit", "-qm", "init"],
        ["git", "checkout", "-q", "-b", "spinwright/test"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    base_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Workspace(
        root=root,
        repo_dir=repo,
        venv_dir=root / ".venv",
        branch="spinwright/test",
        base_sha=base_sha,
        keep=True,
    )


def test_pick_review_attempt_chooses_most_improved_with_a_diff():
    no_diff = _attempt(diff="", rel=None)
    weak = _attempt(diff="diff --git a/m.py b/m.py\n", rel=0.02)
    strong = _attempt(diff="diff --git a/m.py b/m.py\n", rel=0.18)
    assert publish.pick_review_attempt(_loop(no_diff, weak, strong)) is strong


def test_pick_review_attempt_none_when_no_diffs():
    assert publish.pick_review_attempt(_loop(_attempt(diff="", rel=None))) is None


def test_commit_attempt_for_review_applies_and_commits(tmp_path: Path):
    ws = _git_workspace(tmp_path)
    diff = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "diff"],
        capture_output=True,
        text=True,
    ).stdout
    # Produce a real diff, then revert so the tree is clean at base (mirrors
    # the loop's post-reject state).
    (ws.repo_dir / "mod.py").write_text("def x():\n    return 2\n")
    diff = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "diff"], capture_output=True, text=True
    ).stdout
    subprocess.run(
        ["git", "-C", str(ws.repo_dir), "checkout", "--", "mod.py"], check=True
    )
    attempt = _attempt(diff=diff, rel=0.05, reason="walltime 5% below threshold 20%")

    sha = publish.commit_attempt_for_review(ws, attempt)

    assert sha is not None
    assert attempt.commit_sha == sha
    # The change is now committed on the branch.
    assert (ws.repo_dir / "mod.py").read_text() == "def x():\n    return 2\n"
    head_msg = subprocess.run(
        ["git", "-C", str(ws.repo_dir), "log", "-1", "--pretty=%s"],
        capture_output=True,
        text=True,
    ).stdout
    assert "unaccepted attempt for review" in head_msg
    assert "walltime 5%" in head_msg


def test_commit_attempt_for_review_none_when_diff_empty(tmp_path: Path):
    ws = _git_workspace(tmp_path)
    assert publish.commit_attempt_for_review(ws, _attempt(diff="", rel=None)) is None


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
