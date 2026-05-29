from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from spinwright import config as cfg_mod
from spinwright.extraction import extract as extract_mod
from spinwright.llm import client as client_mod
from spinwright.llm.client import ClientProtocol
from spinwright.repo import workspace as workspace_mod


def _load_config(path: str | None) -> cfg_mod.Config:
    if path:
        return cfg_mod.load(path)
    return cfg_mod.default()


def _looks_like_workspace(path: Path) -> bool:
    return (path / ".venv" / "bin" / "python").exists() and (path / "repo" / ".git").exists()


def _resolve_workspace(workspace_arg: str) -> workspace_mod.Workspace:
    """Require an existing prep'd workspace. ``spinwright prep`` is the only
    way to create one — extract/run never implicitly clone. This keeps prep's
    flags (--ref, --extras, --requirements, --workspace) authoritative."""
    candidate = Path(workspace_arg).expanduser()
    if not candidate.exists():
        raise SystemExit(
            f"workspace path {candidate} does not exist.\n"
            "Build a workspace first with: "
            "`spinwright prep <repo_url> --workspace <path> [--requirements ...]`"
        )
    candidate = candidate.resolve()
    if not _looks_like_workspace(candidate):
        raise SystemExit(
            f"{candidate} does not look like a spinwright workspace "
            "(expected .venv/bin/python and repo/.git).\n"
            "Build one with: "
            "`spinwright prep <repo_url> --workspace <path> [--requirements ...]`"
        )
    ws = workspace_mod.reuse(candidate)
    print(f"using workspace at {ws.root}", file=sys.stderr)
    return ws


def _report(result: extract_mod.ExtractionResult) -> None:
    if result.success:
        print()
        print(f"Extraction succeeded for {result.nodeid}")
        print(f"  path:    {result.extraction_path}")
        print(f"  notes:   {result.notes_path}")
        print(f"  commit:  {result.commit_sha}")
        if result.conversation:
            c = result.conversation
            print(f"  tokens:  in={c.input_tokens} out={c.output_tokens} "
                  f"cache_w={c.cache_creation_input_tokens} cache_r={c.cache_read_input_tokens}")
            print(f"  turns:   {len(c.turns)}  stop_reason={c.stop_reason}")
        return

    print()
    print(f"Extraction FAILED for {result.nodeid}")
    print(f"  reason: {result.failure_reason}")
    if result.eligibility_reasons:
        print("  eligibility reasons:")
        for r in result.eligibility_reasons:
            loc = f" (line {r.lineno})" if r.lineno is not None else ""
            print(f"    - [{r.code}] {r.message}{loc}")
    if result.sanity_error:
        print("  sanity error:")
        for line in result.sanity_error.splitlines():
            print(f"    {line}")
    if result.conversation:
        c = result.conversation
        print(f"  conversation: turns={len(c.turns)} stop_reason={c.stop_reason} "
              f"tokens_in={c.input_tokens} tokens_out={c.output_tokens}")


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[], ClientProtocol] = client_mod.make_client,
) -> int:
    cfg = _load_config(args.config)
    ws = _resolve_workspace(args.workspace)

    try:
        client = client_factory()
    except client_mod.MissingAPIKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    result = extract_mod.extract(
        ws=ws,
        nodeid=args.test,
        config=cfg,
        client=client,
    )
    _report(result)
    return 0 if result.success else 1
