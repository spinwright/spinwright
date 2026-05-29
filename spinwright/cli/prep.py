from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinwright import config as cfg_mod
from spinwright.repo import venv, workspace


def _load_config(path: str | None) -> cfg_mod.Config:
    if path:
        return cfg_mod.load(path)
    return cfg_mod.default()


def run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    extras = tuple(e for e in args.extras.split(",") if e) if args.extras else ()
    requirements = tuple(args.requirements or ())

    ref = args.ref or (cfg.target.ref if cfg.target.ref else None)
    print(f"cloning {args.repo!r} (ref={ref or 'HEAD'}) ...", file=sys.stderr)
    ws = workspace.create(
        source=args.repo,
        ref=ref,
        branch_prefix=cfg.pr.branch_prefix,
        branch_suffix="prep",
        keep=True,
    )
    print(f"  workspace: {ws.root}", file=sys.stderr)
    print(f"  base sha:  {ws.base_sha}", file=sys.stderr)
    print(f"  branch:    {ws.branch}", file=sys.stderr)

    print("creating venv + installing target ...", file=sys.stderr)
    venv.create(ws)
    if extras:
        print(f"  extras:        {','.join(extras)}", file=sys.stderr)
    for rel in requirements:
        print(f"  requirements:  {rel}", file=sys.stderr)
    try:
        venv.install_target(ws, extras=extras, requirements_files=requirements)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # stdout: just the workspace path, so it can be captured in shell pipes.
    sys.stdout.write(str(ws.root) + "\n")
    return 0
