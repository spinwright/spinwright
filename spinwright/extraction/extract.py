from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import spinwright
from spinwright.config import Config
from spinwright.extraction import eligibility
from spinwright.llm.client import ClientProtocol
from spinwright.llm.dispatch import ConversationResult, run_conversation
from spinwright.repo import venv as venv_mod
from spinwright.repo import workspace as workspace_mod
from spinwright.repo.workspace import Workspace
from spinwright.tools import process, registry, source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    success: bool
    nodeid: str
    extraction_path: Path | None
    notes_path: Path | None
    commit_sha: str | None
    sanity_passed: bool
    sanity_error: str | None
    conversation: ConversationResult | None
    failure_reason: str | None = None
    eligibility_reasons: list[eligibility.Reason] = field(default_factory=list)


def extract(
    *,
    ws: Workspace,
    nodeid: str,
    config: Config,
    client: ClientProtocol,
    model: str | None = None,
) -> ExtractionResult:
    """Drive the LLM-led extraction conversation for a single test.

    The caller is responsible for confirming eligibility ahead of time; we
    re-check here as a safety net so a buggy caller doesn't write garbage
    extractions. The conversation budget comes from ``config.budget``.

    On success: extraction file written + sanity-checked, NOTES.md written,
    both committed on the working branch. On failure (sanity check fails or
    the conversation never produces the file), nothing is committed.
    """
    test_path = (ws.repo_dir / nodeid.split("::")[0]).resolve()
    if not test_path.exists():
        return _failure(nodeid, f"test source file not found: {test_path}")

    elig = eligibility.check(
        test_path,
        nodeid,
        allow_pure_conftest_imports=config.eligibility.allow_pure_conftest_imports,
    )
    if not elig.eligible:
        return ExtractionResult(
            success=False,
            nodeid=nodeid,
            extraction_path=None,
            notes_path=None,
            commit_sha=None,
            sanity_passed=False,
            sanity_error=None,
            conversation=None,
            failure_reason="test is ineligible for extraction",
            eligibility_reasons=list(elig.reasons),
        )

    test_meta = source.get_test_source(ws.repo_dir, nodeid)
    sanitized_id = sanitize_test_id(nodeid)
    target_path = (ws.repo_dir / config.corpus.dir / f"{sanitized_id}.py").resolve()
    notes_path = target_path.with_suffix(".NOTES.md")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_init_py(target_path.parent)

    tools = registry.build_extraction_tools(
        workspace_root=ws.root,
        repo_dir=ws.repo_dir,
        venv_python=venv_mod.python_executable(ws),
    )
    system_prompt = _build_system_prompt(target_path=target_path)
    user_message = _build_user_message(
        nodeid=nodeid, test_meta=test_meta, target_path=target_path
    )
    chosen_model = model or config.models.model

    conversation = run_conversation(
        client,
        model=chosen_model,
        system=system_prompt,
        initial_user_message=user_message,
        tools=tools,
        max_turns=config.budget.max_extraction_turns,
    )

    if not target_path.exists():
        return ExtractionResult(
            success=False,
            nodeid=nodeid,
            extraction_path=None,
            notes_path=None,
            commit_sha=None,
            sanity_passed=False,
            sanity_error=None,
            conversation=conversation,
            failure_reason=(
                f"conversation ended ({conversation.stop_reason}) without writing "
                f"the extraction file at {target_path}"
            ),
        )

    sanity_ok, sanity_err = _sanity_check(ws, target_path)
    if not sanity_ok:
        return ExtractionResult(
            success=False,
            nodeid=nodeid,
            extraction_path=target_path,
            notes_path=None,
            commit_sha=None,
            sanity_passed=False,
            sanity_error=sanity_err,
            conversation=conversation,
            failure_reason="extraction file did not pass setup/run/verify sanity check",
        )

    _write_notes(notes_path=notes_path, ws=ws, nodeid=nodeid, test_meta=test_meta)
    paths_to_commit = [target_path, notes_path]
    init_py = target_path.parent / "__init__.py"
    if init_py.exists():
        paths_to_commit.append(init_py)
    commit_sha = workspace_mod.commit(
        ws,
        paths_to_commit,
        message=f"spinwright: extract {nodeid}",
    )

    return ExtractionResult(
        success=True,
        nodeid=nodeid,
        extraction_path=target_path,
        notes_path=notes_path,
        commit_sha=commit_sha,
        sanity_passed=True,
        sanity_error=None,
        conversation=conversation,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9]+")


def sanitize_test_id(nodeid: str) -> str:
    """Turn a pytest nodeid into a filesystem-safe identifier.

    ``tests/test_foo.py::TestThing::test_bar`` →
    ``tests_test_foo__TestThing__test_bar`` (without ``.py`` extension).
    Strips parametrize ``[...]`` suffixes since one extraction covers all
    parameter values (parametrized tests are ineligible at the AST stage
    anyway, but the function stays robust).
    """
    rel_path, *parts = nodeid.split("::")
    parts = [p.split("[", 1)[0] for p in parts]
    rel_path = rel_path.removesuffix(".py")
    pieces = [rel_path, *parts]
    return "__".join(_SANITIZE_RE.sub("_", p).strip("_") for p in pieces)


def _ensure_init_py(directory: Path) -> None:
    init = directory / "__init__.py"
    if not init.exists():
        init.write_text("")


def _failure(nodeid: str, reason: str) -> ExtractionResult:
    return ExtractionResult(
        success=False,
        nodeid=nodeid,
        extraction_path=None,
        notes_path=None,
        commit_sha=None,
        sanity_passed=False,
        sanity_error=None,
        conversation=None,
        failure_reason=reason,
    )


def _sanity_check(ws: Workspace, extraction_path: Path) -> tuple[bool, str | None]:
    """Run setup → run → verify once inside the target venv, return (ok, err)."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_sw_extraction', {str(extraction_path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "state = mod.setup()\n"
        "mod.run(state)\n"
        "mod.verify(state)\n"
        "print('SANITY_OK')\n"
    )
    result = process.run_python(
        venv_mod.python_executable(ws),
        code,
        cwd=ws.repo_dir,
    )
    if result.returncode == 0 and "SANITY_OK" in result.stdout:
        return True, None
    err = result.stderr.strip() or result.stdout.strip() or f"rc={result.returncode}"
    return False, err


def _write_notes(
    *, notes_path: Path, ws: Workspace, nodeid: str, test_meta: dict
) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    # Convert the source path to repo-relative so absolute paths (which include
    # /Users/<name>/...) don't end up committed to the target repo's history.
    src_path = Path(test_meta["path"])
    try:
        src_rel = str(src_path.resolve().relative_to(ws.repo_dir.resolve()))
    except ValueError:
        # Source lives outside the repo somehow — keep the basename only, not
        # the absolute path with usernames embedded.
        src_rel = src_path.name
    notes_path.write_text(
        f"# Extraction notes for `{nodeid}`\n\n"
        f"- Source nodeid: `{nodeid}`\n"
        f"- Source path (repo-relative): `{src_rel}`\n"
        f"- Source kind: {test_meta['kind']}\n"
        f"- Source commit SHA: `{ws.base_sha}`\n"
        f"- Extraction date (UTC): {now}\n"
        f"- Spinwright version: {spinwright.__version__}\n"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE = """\
You are Spinwright's extraction agent. Your job is to convert a pytest test \
into a measurement harness with three functions: setup, run, and verify.

The harness will be invoked many times under timeit and (on Linux) Callgrind \
to measure the cost of the operation under test, then once at the end to \
verify correctness. The contract for the three functions is exact:

```python
def setup() -> dict:
    \"\"\"Build inputs and any state required by run(). Called once per
    measurement session. Must be deterministic — seed any RNGs here.
    Pure-Python construction is fine; any I/O is forbidden.\"\"\"

def run(state: dict) -> None:
    \"\"\"The hot path under measurement. Called N times. Must not assert
    or mutate state in a way that changes subsequent calls. Should
    invoke the operation under test, optionally stashing the result on
    state for verify() to inspect.\"\"\"

def verify(state: dict) -> None:
    \"\"\"Correctness check. Called once after the measurement loop.
    Adapts the original test's assertions to read from state.\"\"\"
```

Rules:
- run() must do real work — call into the target package's hot code. Do not
  inline a Python-only re-implementation of what the package does.
- run() must NOT contain assertions or prints.
- setup() must seed any RNG (random.seed / numpy.random.seed) that the
  operation depends on. If neither random is used, no seeding is needed.
- No network, no subprocess, no filesystem access outside tempfile.
- Imports of the target package belong at module top-level, not inside the
  functions.

Workflow:
1. If you need to see other source (e.g., a helper the test calls), use
   read_source(qualname=...).
2. Write the extraction to the EXACT target path provided in the user message
   using write_file. If you need to revise, use edit_file or write_file again.
3. Sanity-check your extraction by calling run_python with a small script that
   does: import the file via importlib.util, call setup(), call run(state),
   call verify(state). The orchestrator will run the same check after you end
   the turn — but checking yourself first saves a round-trip.
4. When you're confident the extraction is correct, end the turn with a brief
   one-sentence confirmation. Do not produce a long summary.

You may end the turn without writing the extraction ONLY if the test cannot be
extracted (e.g., you discover hidden fixture use the AST check missed). In
that case, say so explicitly.
"""


def _build_system_prompt(*, target_path: Path) -> str:
    return _SYSTEM_PROMPT_TEMPLATE


def _build_user_message(*, nodeid: str, test_meta: dict, target_path: Path) -> str:
    lines = [
        f"Extract the test `{nodeid}` into a measurement harness.",
        "",
        f"Write the extraction to this exact path: `{target_path}`",
        "",
        f"Source file: `{test_meta['path']}`",
        f"Lines: {test_meta['lineno']}–{test_meta['end_lineno']}",
        f"Kind: {test_meta['kind']}",
    ]
    if test_meta.get("class_name"):
        lines.append(f"Class: `{test_meta['class_name']}`")
    if test_meta["decorators"]:
        lines.append("Decorators (already cleared by eligibility check):")
        for d in test_meta["decorators"]:
            lines.append(f"  {d}")
    lines.extend(
        [
            "",
            "Test source:",
            "```python",
            test_meta["source"],
            "```",
        ]
    )
    return "\n".join(lines)
