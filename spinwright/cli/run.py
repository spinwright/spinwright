from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable

from spinwright import config as cfg_mod
from spinwright import run_log
from spinwright.cli import extract as cli_extract  # workspace auto-detect / prep
from spinwright.cli._extraction_arg import resolve_extraction
from spinwright.llm import client as client_mod
from spinwright.llm.client import ClientProtocol
from spinwright.optimization import loop as loop_mod
from spinwright.optimization import regression as regression_mod
from spinwright.pr import builder as pr_builder
from spinwright.pr import publish as pr_publish
from spinwright.repo import venv as venv_mod


def _load_config(path: str | None) -> cfg_mod.Config:
    if path:
        return cfg_mod.load(path)
    return cfg_mod.default()


def _progress(msg: str) -> None:
    """Stream live loop/agent progress to stderr. ``flush=True`` so messages
    appear in real time in CI logs even when stdout/stderr is a pipe (the
    GitHub Action pipes ``spinwright run`` through ``tee``)."""
    print(f"[spinwright] {msg}", file=sys.stderr, flush=True)


# Extraction resolution moved to cli._extraction_arg (shared with measure/optimize).


_NODEID_RE = re.compile(r"Source nodeid:\s*`([^`]+)`")
_SHA_RE = re.compile(r"Source commit SHA:\s*`([^`]+)`")


def _read_extraction_metadata(
    extraction_path: Path,
    *,
    fallback_sha: str,
    corpus_dir: str,
) -> pr_builder.ExtractionMetadata:
    """Parse the sibling NOTES.md written by ``spinwright extract``. Missing
    fields fall back to None / the workspace's base SHA."""
    notes = extraction_path.with_suffix(".NOTES.md")
    nodeid = None
    sha = fallback_sha
    if notes.exists():
        text = notes.read_text()
        m = _NODEID_RE.search(text)
        if m:
            nodeid = m.group(1)
        m = _SHA_RE.search(text)
        if m:
            sha = m.group(1)
    return pr_builder.ExtractionMetadata(
        extraction_path=extraction_path,
        original_nodeid=nodeid,
        source_commit_sha=sha,
        corpus_dir=corpus_dir,
    )


def _print_loop_summary(loop_result: loop_mod.LoopResult) -> None:
    print()
    print(f"Agent loop: {loop_result.stop_reason}")
    print(f"  iterations:        {len(loop_result.iterations)}")
    print(f"  accepted patches:  {loop_result.accepted_count}")
    print(f"  explored rejects:  {len(loop_result.explored)}")
    print(f"  tokens spent:      {loop_result.spent_tokens:,}")
    print(f"  wall-clock:        {loop_result.elapsed_seconds:.1f}s")
    if loop_result.baseline is not None:
        b = loop_result.baseline
        print(
            f"  baseline walltime: best={b.walltime.best_seconds * 1e6:.2f} us  "
            f"median={b.walltime.median_seconds * 1e6:.2f} us"
        )
        if b.callgrind is not None:
            print(f"  baseline callgrind: {b.callgrind.instructions:,} inst/call")
    if loop_result.final_walltime is not None:
        f = loop_result.final_walltime
        print(
            f"  final walltime:    best={f.best_seconds * 1e6:.2f} us  "
            f"median={f.median_seconds * 1e6:.2f} us"
        )
    if loop_result.final_callgrind is not None:
        print(
            f"  final callgrind:   {loop_result.final_callgrind.instructions:,} inst/call"
        )
    if loop_result.baseline and loop_result.final_walltime:
        wt_delta = (
            loop_result.baseline.walltime.median_seconds
            - loop_result.final_walltime.median_seconds
        ) / loop_result.baseline.walltime.median_seconds
        print(f"  total walltime delta: {wt_delta:+.2%}")
    if (
        loop_result.baseline
        and loop_result.baseline.callgrind
        and loop_result.final_callgrind
    ):
        cg_delta = (
            loop_result.baseline.callgrind.instructions
            - loop_result.final_callgrind.instructions
        ) / loop_result.baseline.callgrind.instructions
        print(f"  total callgrind delta: {cg_delta:+.2%}")

    for i, it in enumerate(loop_result.iterations):
        label = "ACCEPT" if it.accepted else "reject"
        delta = (
            f"{it.relative_improvement:+.2%}"
            if it.relative_improvement is not None
            else "n/a"
        )
        reason = (
            (it.rejection_reason or "").splitlines()[0] if it.rejection_reason else ""
        )
        print(f"  [{i + 1}] {label}  primary={delta}  reason={reason}")


def _print_regression_summary(reg: regression_mod.RegressionResult) -> None:
    print()
    print("Regression check:")
    print(f"  suite passed:    {reg.passed}")
    print(f"  fallback used:   {reg.fallback_used}")
    print(f"  dropped commits: {len(reg.dropped_commits)}")
    for sha in reg.dropped_commits:
        print(f"    - {sha}")
    if not reg.passed:
        print("  pytest tail:")
        for line in reg.final_pytest_output.splitlines()[-20:]:
            print(f"    {line}")


def _print_publish_summary(pub: pr_publish.PublishResult, run_dir: Path) -> None:
    print()
    print(f"PR mode: {pub.mode}")
    if pub.pr_url:
        print(f"  url: {pub.pr_url}")
    if pub.pushed_branch:
        print(f"  pushed branch: {pub.pushed_branch}")
    if pub.skipped_reason:
        print(f"  note: {pub.skipped_reason}")
    print(f"  PR.md: {run_dir / 'PR.md'}")


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[], ClientProtocol] = client_mod.make_client,
) -> int:
    cfg = _load_config(args.config)
    ws = cli_extract._resolve_workspace(args.workspace)
    extraction = resolve_extraction(ws.root, args.extraction, corpus_dir=cfg.corpus.dir)

    try:
        client = client_factory()
    except client_mod.MissingAPIKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    loop_result = loop_mod.run_loop(
        ws=ws,
        extraction_path=extraction,
        config=cfg,
        client=client,
        on_progress=_progress,
    )
    _print_loop_summary(loop_result)

    if not loop_result.success:
        print(f"\nLoop failed: {loop_result.failure_reason}", file=sys.stderr)
        return 1

    reg: regression_mod.RegressionResult | None = None
    if not args.skip_regression and loop_result.accepted_count > 0:
        reg = regression_mod.run_regression_check(
            ws=ws,
            accepted_commits=[
                loop_result.iterations[i].commit_sha
                for i in loop_result.accepted_indices
                if loop_result.iterations[i].commit_sha
            ],
            venv_python=venv_mod.python_executable(ws),
        )
        _print_regression_summary(reg)

    surviving_patches = loop_result.accepted_count - (
        len(reg.dropped_commits) if reg else 0
    )

    # PR assembly + publish
    run_id = run_log.make_run_id()
    runs_root = Path(args.runs_dir).expanduser()

    pr_draft = None
    publish_result = None
    run_dir = runs_root / run_id  # Default if no_pr path below doesn't set it.
    if not args.no_pr:
        meta = _read_extraction_metadata(
            extraction,
            fallback_sha=ws.base_sha,
            corpus_dir=cfg.corpus.dir,
        )
        pr_draft = pr_builder.build_pr(
            loop_result=loop_result,
            regression=reg,
            extraction=meta,
            run_id=run_id,
            reasoning_model=cfg.models.reasoning,
            repo_dir=ws.repo_dir,
        )
        run_dir = run_log.write_run_directory(
            runs_root=runs_root,
            run_id=run_id,
            pr_draft=pr_draft,
            loop_result=loop_result,
            regression=reg,
            extra_metadata={
                "workspace": str(ws.root),
                "extraction": str(extraction),
                "branch": ws.branch,
                "base_sha": ws.base_sha,
            },
        )
        if surviving_patches > 0:
            publish_result = pr_publish.publish(
                ws=ws,
                pr_draft=pr_draft,
                pr_config=cfg.pr,
                run_dir=run_dir,
            )
            _print_publish_summary(publish_result, run_dir)
        else:
            print(
                f"\nNo surviving patches — PR.md still written to: {run_dir / 'PR.md'}"
            )
    else:
        # Still write a minimal run directory for the summary, no PR.
        run_log.write_run_directory(
            runs_root=runs_root,
            run_id=run_id,
            pr_draft=None,
            loop_result=loop_result,
            regression=reg,
            extra_metadata={
                "workspace": str(ws.root),
                "extraction": str(extraction),
            },
        )

    # Final structured line so callers (CI, scripts) can locate the run dir
    # without scraping the human report. Format: ``RUN_DIR=<absolute path>``.
    print()
    print(f"RUN_DIR={run_dir}")
    print(f"SURVIVING_PATCHES={surviving_patches}")

    if surviving_patches == 0:
        return 1
    return 0 if (reg is None or reg.passed) else 1
