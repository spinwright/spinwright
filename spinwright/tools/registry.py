from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from spinwright.llm.dispatch import ToolDefinition
from spinwright.profiling import cprofile
from spinwright.tools import edit, git, process, source


def build_extraction_tools(
    *,
    workspace_root: Path,
    repo_dir: Path,
    venv_python: Path,
    run_python_timeout: float = 30.0,
) -> list[ToolDefinition]:
    """Bind workspace context into the minimum tool set the extraction
    conversation needs (SPEC §8.1, §8.2 subset, §8.5, §8.6 subset)."""

    def _list_tests(args: dict) -> list[str]:
        return source.list_tests(
            venv_python=venv_python,
            repo_dir=repo_dir,
            pattern=args.get("pattern"),
        )

    def _get_test_source(args: dict) -> dict:
        return source.get_test_source(repo_dir=repo_dir, nodeid=args["nodeid"])

    def _read_source(args: dict) -> dict:
        return source.read_source(venv_python=venv_python, qualname=args["qualname"])

    def _write_file(args: dict) -> dict:
        result = edit.write_file(
            workspace_root=workspace_root,
            path=args["path"],
            content=args["content"],
        )
        return asdict(result)

    def _edit_file(args: dict) -> dict:
        result = edit.edit_file(
            workspace_root=workspace_root,
            path=args["path"],
            old_string=args["old_string"],
            new_string=args["new_string"],
        )
        return asdict(result)

    def _run_python(args: dict) -> dict:
        result = process.run_python(
            venv_python=venv_python,
            code=args["code"],
            cwd=repo_dir,
            timeout_seconds=args.get("timeout_seconds", run_python_timeout),
        )
        return asdict(result)

    return [
        ToolDefinition(
            name="list_tests",
            description=(
                "Enumerate pytest test nodeids in the target repository. "
                "Optionally filter by name pattern via the `-k` expression."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Optional pytest `-k` expression.",
                    },
                },
                "additionalProperties": False,
            },
            handler=_list_tests,
        ),
        ToolDefinition(
            name="get_test_source",
            description=(
                "Return the source code, line range, and decorators for a "
                "single test identified by its pytest nodeid."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "nodeid": {
                        "type": "string",
                        "description": "Pytest nodeid (e.g., tests/test_foo.py::test_bar).",
                    },
                },
                "required": ["nodeid"],
                "additionalProperties": False,
            },
            handler=_get_test_source,
        ),
        ToolDefinition(
            name="read_source",
            description=(
                "Resolve a dotted Python qualname to its source file, line "
                "number, and source code. Runs `inspect.getsourcelines` "
                "inside the target venv."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "qualname": {
                        "type": "string",
                        "description": "e.g., static_frame.core.index.Index._loc_to_iloc",
                    },
                },
                "required": ["qualname"],
                "additionalProperties": False,
            },
            handler=_read_source,
        ),
        ToolDefinition(
            name="write_file",
            description=(
                "Create or overwrite a file inside the workspace. Path must "
                "be relative to the workspace root or absolute and inside it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_write_file,
        ),
        ToolDefinition(
            name="edit_file",
            description=(
                "Replace `old_string` with `new_string` inside an existing "
                "workspace file. `old_string` must occur exactly once — "
                "include surrounding context if needed to make it unique."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
            handler=_edit_file,
        ),
        ToolDefinition(
            name="run_python",
            description=(
                "Run a short Python snippet via the target venv's Python. "
                "Use for import sanity checks, quick prints, or running an "
                "extraction module's setup/run/verify by hand. Captures "
                "stdout/stderr and returncode."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source executed with `python -c`.",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Override the default 30s timeout.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            handler=_run_python,
        ),
    ]


def build_optimization_tools(
    *,
    workspace_root: Path,
    repo_dir: Path,
    venv_python: Path,
    extraction_path: Path,
    run_python_timeout: float = 30.0,
    profile_default_iterations: int = 1000,
) -> list[ToolDefinition]:
    """Tools available during the single-shot optimization conversation.

    Wraps the read/edit subset of the extraction toolset plus profiling and
    git-revert. The extraction path is closed over so the LLM doesn't have to
    pass it on every profile call.
    """

    def _read_source(args: dict) -> dict:
        return source.read_source(venv_python=venv_python, qualname=args["qualname"])

    def _edit_file(args: dict) -> dict:
        result = edit.edit_file(
            workspace_root=workspace_root,
            path=args["path"],
            old_string=args["old_string"],
            new_string=args["new_string"],
        )
        return asdict(result)

    def _write_file(args: dict) -> dict:
        result = edit.write_file(
            workspace_root=workspace_root,
            path=args["path"],
            content=args["content"],
        )
        return asdict(result)

    def _run_python(args: dict) -> dict:
        result = process.run_python(
            venv_python=venv_python,
            code=args["code"],
            cwd=repo_dir,
            timeout_seconds=args.get("timeout_seconds", run_python_timeout),
        )
        return asdict(result)

    def _profile_cprofile(args: dict) -> dict:
        excludes = tuple(args.get("exclude_paths", ()))
        iterations = args.get("iterations", profile_default_iterations)
        sort_by = args.get("sort_by", "cumtime")
        limit = args.get("limit", 25)
        result = cprofile.profile_cprofile(
            venv_python,
            extraction_path,
            iterations=iterations,
            exclude_paths=excludes,
        )
        top = cprofile.top_entries(result, by=sort_by, limit=limit)
        return {
            "iterations": result.iterations,
            "total_seconds": result.total_seconds,
            "verify_passed": result.verify_passed,
            "verify_error": result.verify_error,
            "sort_by": sort_by,
            "entries": [asdict(e) for e in top],
        }

    def _git_diff(args: dict) -> str:
        return git.git_diff(repo_dir)

    def _git_revert_path(args: dict) -> dict:
        result = git.git_revert_path(
            workspace_root=workspace_root,
            repo_dir=repo_dir,
            path=args["path"],
        )
        return asdict(result)

    def _git_revert_all(args: dict) -> dict:
        result = git.git_revert_all(repo_dir=repo_dir)
        return asdict(result)

    return [
        ToolDefinition(
            name="profile_cprofile",
            description=(
                "Profile the extraction's run() under cProfile. Returns the "
                "hottest functions sorted by `sort_by` (default cumtime). "
                "Use `exclude_paths` to drop noise (e.g. stdlib path)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "iterations": {"type": "integer", "description": "How many run() calls to profile."},
                    "exclude_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Substring-match these against source filenames; matching entries are dropped.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["tottime", "cumtime", "tottime_per_call", "cumtime_per_call"],
                    },
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            handler=_profile_cprofile,
        ),
        ToolDefinition(
            name="read_source",
            description="Resolve a dotted qualname to its source code, file, and line range.",
            input_schema={
                "type": "object",
                "properties": {"qualname": {"type": "string"}},
                "required": ["qualname"],
                "additionalProperties": False,
            },
            handler=_read_source,
        ),
        ToolDefinition(
            name="edit_file",
            description=(
                "Replace `old_string` with `new_string` in a workspace file. "
                "`old_string` must occur exactly once."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
            handler=_edit_file,
        ),
        ToolDefinition(
            name="write_file",
            description="Create or overwrite a file inside the workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_write_file,
        ),
        ToolDefinition(
            name="run_python",
            description="Run a short Python snippet via the target venv's Python.",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            handler=_run_python,
        ),
        ToolDefinition(
            name="git_diff",
            description="Return the current working-tree diff against HEAD.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_git_diff,
        ),
        ToolDefinition(
            name="git_revert_path",
            description=(
                "Restore one workspace file to its HEAD state, discarding "
                "uncommitted changes to it."
            ),
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_git_revert_path,
        ),
        ToolDefinition(
            name="git_revert_all",
            description=(
                "Restore every tracked file in the repo to HEAD. Untracked "
                "files (e.g. newly created extractions) are left in place."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_git_revert_all,
        ),
    ]
