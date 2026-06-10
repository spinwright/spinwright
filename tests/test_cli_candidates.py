from __future__ import annotations

import argparse
import json as json_mod
import subprocess
import sys
import textwrap
from pathlib import Path

from spinwright.cli import candidates as cli_candidates
from spinwright.repo.workspace import Workspace


# ---------------------------------------------------------------------------
# Workspace + venv shim shared by both tests
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    repo = root / "repo"
    venv = root / ".venv"
    # Symlink the whole .venv to the spinwright dev venv root so subprocess
    # invocations through .venv/bin/python actually find pytest.
    root.mkdir(parents=True)
    venv.symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_seed.py").write_text("def test_init(): assert True\n")
    for cmd in (
        ["git", "init", "-b", "main"],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "add", "."],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "commit", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    base_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-b", "spinwright/test"],
        check=True,
        capture_output=True,
    )
    return Workspace(
        root=root,
        repo_dir=repo,
        venv_dir=venv,
        branch="spinwright/test",
        base_sha=base_sha,
        keep=True,
    )


def _add_mixed_tests(ws: Workspace) -> None:
    """A slow eligible test, a slow ineligible (fixture-using) test, and a
    fast test that shouldn't surface as a candidate."""
    (ws.repo_dir / "tests" / "test_speeds.py").write_text(
        textwrap.dedent("""
        import time
        import pytest

        def test_slow_eligible():
            time.sleep(0.2)
            assert sum(i * i for i in range(10)) == 285

        def test_slower_but_ineligible(my_fixture):
            time.sleep(0.3)
            assert my_fixture is not None

        @pytest.fixture
        def my_fixture():
            return object()

        def test_fast():
            assert True
    """).lstrip("\n")
    )
    subprocess.run(
        ["git", "-C", str(ws.repo_dir), "add", "."], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(ws.repo_dir),
            "-c",
            "user.email=x@x",
            "-c",
            "user.name=x",
            "commit",
            "-m",
            "mixed",
        ],
        check=True,
        capture_output=True,
    )


def _args(ws: Workspace, **overrides) -> argparse.Namespace:
    defaults = dict(
        workspace=str(ws.root),
        test_path=[],
        json=False,
        nodeids=False,
        limit=50,
        config=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_candidates_lists_eligible_and_rejected_sections(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    _add_mixed_tests(ws)
    cfg_path = tmp_path / "spinwright.toml"
    cfg_path.write_text("[test_selection]\nslow_threshold_seconds = 0.05\n")

    rc = cli_candidates.run(_args(ws, config=str(cfg_path)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Eligible candidates" in out
    assert "test_slow_eligible" in out
    assert "Rejected candidates" in out
    assert "test_slower_but_ineligible" in out


def test_candidates_test_path_constrains_discovery(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    (ws.repo_dir / "included" / "tests").mkdir(parents=True)
    (ws.repo_dir / "included" / "tests" / "test_a.py").write_text(
        textwrap.dedent("""
        import time
        def test_a_slow():
            time.sleep(0.2)
            assert True
    """).lstrip("\n")
    )
    (ws.repo_dir / "excluded" / "tests").mkdir(parents=True)
    (ws.repo_dir / "excluded" / "tests" / "test_b.py").write_text(
        textwrap.dedent("""
        import time
        def test_b_slow():
            time.sleep(0.2)
            assert True
    """).lstrip("\n")
    )
    subprocess.run(
        ["git", "-C", str(ws.repo_dir), "add", "."], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(ws.repo_dir),
            "-c",
            "user.email=x@x",
            "-c",
            "user.name=x",
            "commit",
            "-m",
            "two trees",
        ],
        check=True,
        capture_output=True,
    )

    cfg_path = tmp_path / "spinwright.toml"
    cfg_path.write_text("[test_selection]\nslow_threshold_seconds = 0.05\n")

    rc = cli_candidates.run(_args(ws, test_path=["included"], config=str(cfg_path)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "test_a_slow" in out
    assert "test_b_slow" not in out


def test_candidates_json_emits_machine_readable(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    _add_mixed_tests(ws)
    cfg_path = tmp_path / "spinwright.toml"
    cfg_path.write_text("[test_selection]\nslow_threshold_seconds = 0.05\n")

    rc = cli_candidates.run(_args(ws, json=True, config=str(cfg_path)))
    out = capsys.readouterr().out
    assert rc == 0
    payload = json_mod.loads(out)
    assert "candidates" in payload
    nodeids = [c["nodeid"] for c in payload["candidates"]]
    assert any("test_slow_eligible" in n for n in nodeids)
    eligibility_map = {c["nodeid"]: c["eligible"] for c in payload["candidates"]}
    eligible_one = next(n for n in eligibility_map if "test_slow_eligible" in n)
    assert eligibility_map[eligible_one] is True
    ineligible_one = next(
        n for n in eligibility_map if "test_slower_but_ineligible" in n
    )
    assert eligibility_map[ineligible_one] is False


def test_candidates_nodeids_prints_eligible_only(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    _add_mixed_tests(ws)
    cfg_path = tmp_path / "spinwright.toml"
    cfg_path.write_text("[test_selection]\nslow_threshold_seconds = 0.05\n")

    rc = cli_candidates.run(_args(ws, nodeids=True, config=str(cfg_path)))
    out = capsys.readouterr().out
    assert rc == 0
    lines = [line for line in out.splitlines() if line.strip()]
    # All emitted lines are eligible nodeids; ineligible ones are filtered out
    assert all("::" in line for line in lines)
    assert any("test_slow_eligible" in line for line in lines)
    assert not any("test_slower_but_ineligible" in line for line in lines)
