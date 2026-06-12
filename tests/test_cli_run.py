from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinwright.cli import run as cli_run
from spinwright.repo.workspace import Workspace


# --- compact FakeClient (same shape used by other CLI tests) ---


@dataclass
class FakeText:
    text: str
    type: str = "text"

    def model_dump(self, *, exclude_none=True):
        return {"type": "text", "text": self.text}


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def model_dump(self, *, exclude_none=True):
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def model_dump(self, *, exclude_none=True):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self):
        self.responses, self.calls = [], []

    def queue(self, *r):
        self.responses.extend(r)

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        if not self.responses:
            raise AssertionError("no fake response queued")
        return self.responses.pop(0)


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.messages = FakeMessages()

    def create_message(self, **kwargs):
        from spinwright.llm.providers.base import ProviderResponse

        self.messages.calls.append(copy.deepcopy(kwargs))
        if not self.messages.responses:
            raise AssertionError("no fake response queued for this call")
        fake = self.messages.responses.pop(0)
        content = [b.model_dump() for b in fake.content]
        usage = (
            fake.usage.model_dump()
            if hasattr(fake.usage, "model_dump")
            else (fake.usage or {})
        )
        return ProviderResponse(
            content=content, stop_reason=fake.stop_reason, usage=usage
        )


FakeClient = FakeProvider


_TARGET = """
import time

def slow(n):
    time.sleep(0.003)
    return sum(i * i for i in range(n))
"""

_SUITE = """
from target_pkg import slow

def test_slow():
    assert slow(10) == sum(i * i for i in range(10))
"""

_EXTRACTION = """
from target_pkg import slow

def setup():
    return {"n": 200}

def run(state):
    state["_out"] = slow(state["n"])

def verify(state):
    assert state["_out"] == sum(i * i for i in range(state["n"]))
"""


def _make_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    root = tmp_path / "ws"
    root.mkdir(parents=True)
    repo = root / "repo"
    # Symlink the whole ``.venv`` to the spinwright dev venv root so the
    # workspace passes auto-detection AND so subprocess invocations through
    # ``.venv/bin/python`` have pytest available (a simple `bin/python` symlink
    # chases through to base homebrew Python via pyvenv resolution and loses
    # the package set). Real ``spinwright prep`` builds a proper per-workspace
    # venv; the test takes the shortcut of pointing at the dev venv directly.
    venv = root / ".venv"
    venv.symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    (repo / "target_pkg").mkdir(parents=True)
    (repo / "target_pkg" / "__init__.py").write_text(
        textwrap.dedent(_TARGET).lstrip("\n")
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_target.py").write_text(textwrap.dedent(_SUITE).lstrip("\n"))
    extractions = repo / "spinwright"
    extractions.mkdir(parents=True)
    (extractions / "__init__.py").write_text("")
    ext_path = extractions / "demo.py"
    ext_path.write_text(textwrap.dedent(_EXTRACTION).lstrip("\n"))
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
    ), ext_path


# def test_run_cli_drives_loop_and_regression(tmp_path: Path, capsys):
#     ws, ext_path = _make_workspace(tmp_path)
#     target_file = str((ws.repo_dir / "target_pkg" / "__init__.py").resolve())

#     client = FakeClient()
#     # One accepted iteration that removes the sleep, then the loop hits the cap.
#     client.messages.queue(
#         FakeMessage(
#             content=[
#                 FakeToolUse(
#                     id="tu_1",
#                     name="edit_file",
#                     input={
#                         "path": target_file,
#                         "old_string": "time.sleep(0.003)\n    ",
#                         "new_string": "",
#                     },
#                 )
#             ],
#             stop_reason="tool_use",
#         ),
#         FakeMessage(content=[FakeText(text="killed sleep")], stop_reason="end_turn"),
#     )

#     cfg_path = tmp_path / "spinwright.toml"
#     cfg_path.write_text(
#         textwrap.dedent("""
#         [budget]
#         max_patches_proposed = 1
#         max_extraction_turns = 4

#         [measurement]
#         walltime_repeats = 3
#     """)
#     )
#     args = argparse.Namespace(
#         workspace=str(ws.root),
#         extraction=str(ext_path),
#         config=str(cfg_path),
#         skip_regression=False,
#         no_pr=False,
#         runs_dir=str(tmp_path / "runs"),
#         model=None,
#     )

#     rc = cli_run.run(args, provider_factory=lambda _spec: (client, "claude-test"))
#     out = capsys.readouterr().out
#     assert "Agent loop" in out
#     assert "Regression check" in out
#     assert "suite passed:    True" in out
#     # The accepted patch survived regression
#     assert "dropped commits: 0" in out
#     assert "PR mode" in out
#     # PR.md was written to the run dir
#     run_dirs = list((tmp_path / "runs").iterdir())
#     assert len(run_dirs) == 1
#     assert (run_dirs[0] / "PR.md").exists()
#     pr_md = (run_dirs[0] / "PR.md").read_text()
#     assert "## Summary" in pr_md
#     assert "## Measurements" in pr_md
#     assert "## Bottlenecks and Changes" in pr_md
#     assert rc == 0


# def test_run_cli_drops_regressing_patch(tmp_path: Path, capsys):
#     """LLM 'optimizes' by changing slow() to return 0 — passes verify in the
#     extraction (no, wait — verify checks the value, so verify fails). Need a
#     case where the extraction's own verify is fine but the broader suite
#     catches the regression.

#     Setup: extraction's verify is tolerant (no value check); the suite checks
#     the exact return value. The LLM makes an edit that breaks the suite but
#     not verify, then we expect the regression check to drop the commit."""
#     ws, ext_path = _make_workspace(tmp_path)
#     # Loosen the extraction's verify so the LLM can break the suite without
#     # tripping verify.
#     ext_path.write_text(
#         textwrap.dedent("""
#         from target_pkg import slow

#         def setup():
#             return {"n": 10}

#         def run(state):
#             state["_out"] = slow(state["n"])

#         def verify(state):
#             # Intentionally lax — just check we got an int.
#             assert isinstance(state["_out"], int)
#     """).lstrip("\n")
#     )
#     subprocess.run(
#         ["git", "-C", str(ws.repo_dir), "add", "."], check=True, capture_output=True
#     )
#     subprocess.run(
#         [
#             "git",
#             "-C",
#             str(ws.repo_dir),
#             "-c",
#             "user.email=x@x",
#             "-c",
#             "user.name=x",
#             "commit",
#             "--amend",
#             "--no-edit",
#         ],
#         check=True,
#         capture_output=True,
#     )
#     new_base = subprocess.run(
#         ["git", "-C", str(ws.repo_dir), "rev-parse", "HEAD"],
#         check=True,
#         capture_output=True,
#         text=True,
#     ).stdout.strip()
#     ws_with_new_base = Workspace(
#         root=ws.root,
#         repo_dir=ws.repo_dir,
#         venv_dir=ws.venv_dir,
#         branch=ws.branch,
#         base_sha=new_base,
#         keep=True,
#     )

#     target_file = str((ws.repo_dir / "target_pkg" / "__init__.py").resolve())

#     client = FakeClient()
#     # LLM "optimizes" by returning a constant — verify still passes (lax),
#     # but the suite's exact-value check will fail.
#     client.messages.queue(
#         FakeMessage(
#             content=[
#                 FakeToolUse(
#                     id="tu_1",
#                     name="edit_file",
#                     input={
#                         "path": target_file,
#                         "old_string": "    time.sleep(0.003)\n    return sum(i * i for i in range(n))",
#                         "new_string": "    return 0",
#                     },
#                 )
#             ],
#             stop_reason="tool_use",
#         ),
#         FakeMessage(content=[FakeText(text="constant!")], stop_reason="end_turn"),
#     )

#     cfg_path = tmp_path / "spinwright.toml"
#     cfg_path.write_text(
#         textwrap.dedent("""
#         [budget]
#         max_patches_proposed = 1
#         max_extraction_turns = 4

#         [measurement]
#         walltime_repeats = 3
#     """)
#     )
#     args = argparse.Namespace(
#         workspace=str(ws_with_new_base.root),
#         extraction=str(ext_path),
#         config=str(cfg_path),
#         skip_regression=False,
#         no_pr=False,
#         runs_dir=str(tmp_path / "runs"),
#         model=None,
#     )

#     rc = cli_run.run(args, provider_factory=lambda _spec: (client, "claude-test"))
#     out = capsys.readouterr().out
#     assert "Regression check" in out
#     # Patch was dropped because it broke the suite
#     assert "dropped commits: 1" in out
#     # Final accepted count is zero
#     assert rc == 1


# def test_run_cli_handles_missing_api_key(tmp_path: Path, capsys):
#     ws, ext_path = _make_workspace(tmp_path)

#     def boom():
#         raise cli_run.factory_mod.MissingAPIKeyError("no key")

#     args = argparse.Namespace(
#         workspace=str(ws.root),
#         extraction=str(ext_path),
#         config=None,
#         skip_regression=True,
#         no_pr=False,
#         runs_dir=str(tmp_path / "runs"),
#         model=None,
#     )
#     rc = cli_run.run(args, provider_factory=boom)
#     err = capsys.readouterr().err
#     assert rc == 2
#     assert "no key" in err
