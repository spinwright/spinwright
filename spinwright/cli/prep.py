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
    venv.install_target(ws, extras=extras)

    # stdout: just the workspace path, so it can be captured in shell pipes.
    sys.stdout.write(str(ws.root) + "\n")
    return 0
