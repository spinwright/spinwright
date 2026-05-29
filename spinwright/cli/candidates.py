from __future__ import annotations

import argparse
import json
import sys

from spinwright import config as cfg_mod
from spinwright.cli import extract as cli_extract  # workspace resolver only
from spinwright.extraction import discovery, eligibility
from spinwright.repo import venv as venv_mod


def _load_config(path: str | None) -> cfg_mod.Config:
    if path:
        return cfg_mod.load(path)
    return cfg_mod.default()


def _scan(
    ws, cfg: cfg_mod.Config, *, test_paths: tuple[str, ...] = (),
) -> tuple[list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]],
           discovery.DiscoveryReport]:
    scope = ", ".join(test_paths) if test_paths else "full suite"
    print(f"discovering slow tests ({scope}, with --durations=0) ...", file=sys.stderr)
    report = discovery.discover_verbose(
        venv_python=venv_mod.python_executable(ws),
        repo_dir=ws.repo_dir,
        slow_threshold_seconds=cfg.test_selection.slow_threshold_seconds,
        pytest_paths=test_paths,
    )
    print(f"  pytest exit code: {report.returncode}", file=sys.stderr)
    print(f"  found {len(report.durations)} test(s) at or above "
          f"{cfg.test_selection.slow_threshold_seconds}s", file=sys.stderr)
    results: list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]] = []
    for nd in report.durations:
        test_file = ws.repo_dir / nd.nodeid.split("::")[0]
        if not test_file.exists():
            continue
        r = eligibility.check(
            test_file, nd.nodeid,
            allow_pure_conftest_imports=cfg.eligibility.allow_pure_conftest_imports,
        )
        results.append((nd, r))
    return results, report


def _print_human(
    candidates: list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]],
    *, limit: int = 50,
) -> None:
    eligible = [(nd, r) for nd, r in candidates if r.eligible]
    rejected = [(nd, r) for nd, r in candidates if not r.eligible]
    print()
    print(f"Eligible candidates ({len(eligible)}):")
    for nd, _ in eligible[:limit]:
        print(f"  {nd.seconds:7.3f}s   {nd.nodeid}")
    if len(eligible) > limit:
        print(f"  ... ({len(eligible) - limit} more)")
    print()
    print(f"Rejected candidates ({len(rejected)}):")
    for nd, r in rejected[:limit]:
        codes = ",".join(sorted({reason.code for reason in r.reasons}))
        print(f"  {nd.seconds:7.3f}s   {nd.nodeid}  [{codes}]")
    if len(rejected) > limit:
        print(f"  ... ({len(rejected) - limit} more)")


def _print_nodeids_only(
    candidates: list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]],
) -> None:
    """Print one eligible nodeid per line, sorted by duration desc. Suitable
    for piping into `xargs -I {} spinwright extract <ws> --test {}` or similar."""
    for nd, r in candidates:
        if r.eligible:
            print(nd.nodeid)


def _print_json(
    candidates: list[tuple[discovery.NodeDuration, eligibility.EligibilityResult]],
    report: discovery.DiscoveryReport,
) -> None:
    payload = {
        "pytest_returncode": report.returncode,
        "candidates": [
            {
                "nodeid": nd.nodeid,
                "duration_seconds": nd.seconds,
                "eligible": r.eligible,
                "rejection_codes": sorted({reason.code for reason in r.reasons}),
            }
            for nd, r in candidates
        ],
    }
    print(json.dumps(payload, indent=2))


def run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    ws = cli_extract._resolve_workspace(args.workspace)
    candidates, report = _scan(ws, cfg, test_paths=tuple(args.test_path))

    if not report.durations and report.returncode != 0:
        # Surface collection errors so the user knows what to fix.
        print("  pytest stdout tail:", file=sys.stderr)
        for line in report.stdout_tail.splitlines():
            print(f"    {line}", file=sys.stderr)
        if report.stderr_tail:
            print("  pytest stderr tail:", file=sys.stderr)
            for line in report.stderr_tail.splitlines():
                print(f"    {line}", file=sys.stderr)
        if report.returncode == 5:
            print("  → pytest collected zero tests. Check --test-path or "
                  "required test deps.", file=sys.stderr)
        else:
            print("  → collection likely errored above. Install missing test "
                  "deps via `spinwright prep ... --requirements <file>` and "
                  "rebuild the workspace.", file=sys.stderr)

    if args.json:
        _print_json(candidates, report)
    elif args.nodeids:
        _print_nodeids_only(candidates)
    else:
        _print_human(candidates, limit=args.limit)
    return 0
