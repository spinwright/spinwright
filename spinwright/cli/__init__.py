from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spinwright")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser(
        "prep",
        help="Clone a repo into a workspace and install it in a fresh venv.",
    )
    p_prep.add_argument("repo", help="Path or URL of the target repo.")
    p_prep.add_argument("--ref", default=None, help="Git ref to check out (default: clone HEAD).")
    p_prep.add_argument(
        "--extras",
        default="",
        help="Comma-separated optional extras to install (e.g. 'dev,test').",
    )
    p_prep.add_argument(
        "--requirements", "-r",
        action="append",
        default=[],
        metavar="REPO_RELATIVE_PATH",
        help=(
            "Path (repo-relative) to a requirements or lock file to install "
            "after the editable install. Repeatable. E.g. "
            "`--requirements requirements-dev.txt -r requirements/test.txt`."
        ),
    )
    p_prep.add_argument("--config", default=None, help="Path to spinwright.toml.")

    p_extract = sub.add_parser(
        "extract",
        help="Extract a test into a measurement harness.",
        description=(
            "Run the LLM-driven extraction agent against one test. The repo arg "
            "can be either a URL/path to clone (a fresh workspace is prepped) or "
            "a path to an existing `spinwright prep` workspace (reused in place). "
            "If --test is omitted, the slowest eligible test from the target's "
            "pytest suite is selected automatically (SPEC §5.1)."
        ),
    )
    p_extract.add_argument("repo", help="Path or URL of the target repo, OR an existing workspace path.")
    p_extract.add_argument(
        "--test", default=None,
        help="Pytest nodeid to extract. If omitted, the slowest eligible test is auto-selected.",
    )
    p_extract.add_argument(
        "--list-candidates", action="store_true",
        help="Print discovered slow + eligible tests and exit without extracting.",
    )
    p_extract.add_argument("--config", default=None, help="Path to spinwright.toml.")

    p_measure = sub.add_parser("measure", help="Measure an extracted harness.")
    p_measure.add_argument("workspace", help="Path to a workspace built by `spinwright prep`.")
    p_measure.add_argument("--extraction", required=True, help="Path to extraction module.")
    p_measure.add_argument("--config", default=None, help="Path to spinwright.toml.")
    p_measure.add_argument("--repeats", type=int, default=None,
                           help="Override config measurement.walltime_repeats.")
    p_measure.add_argument("--json", action="store_true",
                           help="Emit the result as a JSON object on stdout (for scripting).")

    p_run = sub.add_parser(
        "run",
        help="Full agent loop: profile, optimize, repeat, then regression-check.",
        description=(
            "End-to-end M3 pipeline against an already-extracted harness. "
            "Auto-detects an existing workspace or preps a fresh one. "
            "Iterates the LLM optimization agent up to budget.max_patches_proposed times, "
            "then runs the full pytest suite and drops any patch that breaks tests."
        ),
    )
    p_run.add_argument("repo", help="Path or URL of the target repo, OR an existing workspace path.")
    p_run.add_argument("--extraction", required=True, help="Path to extraction module.")
    p_run.add_argument("--config", default=None, help="Path to spinwright.toml.")
    p_run.add_argument(
        "--exclude-path", action="append", default=[],
        help="Substring excluded from profile output (repeatable).",
    )
    p_run.add_argument(
        "--skip-regression", action="store_true",
        help="Skip the full-suite regression check at the end of the loop.",
    )
    p_run.add_argument(
        "--no-pr", action="store_true",
        help="Skip the PR assembly + publish step (still writes the run directory).",
    )
    p_run.add_argument(
        "--runs-dir", default="./spinwright-runs",
        help="Directory for per-run artifact subdirectories.",
    )

    p_optimize = sub.add_parser(
        "optimize",
        help="Run one round of LLM-driven optimization on an extracted harness.",
        description=(
            "Measure the extraction, drive the optimization agent for one pass, "
            "remeasure, accept the edit if median wallclock improved by at least "
            "the configured threshold (default 20 percent) AND verify still passes, "
            "otherwise revert."
        ),
    )
    p_optimize.add_argument("workspace", help="Path to a workspace built by `spinwright prep` or `spinwright extract`.")
    p_optimize.add_argument("--extraction", required=True, help="Path to extraction module.")
    p_optimize.add_argument("--config", default=None, help="Path to spinwright.toml.")
    p_optimize.add_argument(
        "--exclude-path",
        action="append",
        default=[],
        help="Substring excluded from profile output (repeatable).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prep":
        from spinwright.cli import prep
        return prep.run(args)
    if args.command == "extract":
        from spinwright.cli import extract
        return extract.run(args)
    if args.command == "measure":
        from spinwright.cli import measure
        return measure.run(args)
    if args.command == "optimize":
        from spinwright.cli import optimize
        return optimize.run(args)
    if args.command == "run":
        from spinwright.cli import run
        return run.run(args)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
