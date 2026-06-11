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
    p_prep.add_argument(
        "--workspace",
        "-w",
        default=None,
        metavar="PATH",
        help=(
            "Explicit workspace path. Created if it doesn't exist; must be "
            "empty if it does. If omitted, a tempfile.mkdtemp() under TMPDIR "
            "is used. Handy for local iteration: set this to a stable path "
            "and re-target the same workspace across runs."
        ),
    )
    p_prep.add_argument(
        "--ref", default=None, help="Git ref to check out (default: clone HEAD)."
    )
    p_prep.add_argument(
        "--extras",
        default="",
        help="Comma-separated optional extras to install (e.g. 'dev,test').",
    )
    p_prep.add_argument(
        "--requirements",
        "-r",
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

    p_candidates = sub.add_parser(
        "candidates",
        help="Discover slow + eligible tests in a workspace.",
        description=(
            "Run pytest with --durations=0 in the workspace's venv, then run "
            "AST eligibility on each slow test. Prints the candidate list so "
            "you can pick a test to feed into `spinwright extract --test`. "
            "Does not write or modify anything."
        ),
    )
    p_candidates.add_argument(
        "workspace", help="Path to a workspace built by `spinwright prep`."
    )
    p_candidates.add_argument(
        "--test-path",
        action="append",
        default=[],
        metavar="REPO_RELATIVE_PATH",
        help=(
            "Constrain discovery to a subset of the test tree. Passed as a "
            "positional pytest argument; repeatable."
        ),
    )
    p_candidates.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON document instead of the human-readable list.",
    )
    p_candidates.add_argument(
        "--nodeids",
        action="store_true",
        help=(
            "Print eligible nodeids only, one per line, sorted by duration "
            "desc. Pipes cleanly into other tools."
        ),
    )
    p_candidates.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max entries per section in human output (default: 50).",
    )
    p_candidates.add_argument("--config", default=None, help="Path to spinwright.toml.")

    p_extract = sub.add_parser(
        "extract",
        help="Extract one specific test into a measurement harness.",
        description=(
            "Run the LLM-driven extraction agent against one test in an existing "
            "workspace. Use `spinwright candidates` first to find a test to feed "
            "in via --test."
        ),
    )
    p_extract.add_argument(
        "workspace", help="Path to a workspace built by `spinwright prep`."
    )
    p_extract.add_argument(
        "--test",
        required=True,
        help="Pytest nodeid to extract (use `spinwright candidates` to find one).",
    )
    p_extract.add_argument("--config", default=None, help="Path to spinwright.toml.")

    p_measure = sub.add_parser("measure", help="Measure an extracted harness.")
    p_measure.add_argument(
        "workspace", help="Path to a workspace built by `spinwright prep`."
    )
    p_measure.add_argument(
        "--extraction",
        required=True,
        metavar="NAME",
        help="Extraction name (stem of the file under <repo>/<corpus_dir>/). NAME, NAME.py, or <corpus_dir>/NAME.py all work.",
    )
    p_measure.add_argument("--config", default=None, help="Path to spinwright.toml.")
    p_measure.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override config measurement.walltime_repeats.",
    )
    p_measure.add_argument(
        "--json",
        action="store_true",
        help="Emit the result as a JSON object on stdout (for scripting).",
    )

    p_run = sub.add_parser(
        "run",
        help="Full agent loop: profile, optimize, repeat, then regression-check.",
        description=(
            "End-to-end M3 pipeline against an already-extracted harness in an "
            "existing workspace. Build the workspace with `spinwright prep`. "
            "Iterates the LLM optimization agent up to budget.max_patches_proposed "
            "times, then runs the full pytest suite and drops any patch that "
            "breaks tests."
        ),
    )
    p_run.add_argument(
        "workspace", help="Path to a workspace built by `spinwright prep`."
    )
    p_run.add_argument(
        "--extraction",
        required=True,
        metavar="NAME",
        help="Extraction name; see `spinwright measure --help` for accepted forms.",
    )
    p_run.add_argument("--config", default=None, help="Path to spinwright.toml.")
    p_run.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip the full-suite regression check at the end of the loop.",
    )
    p_run.add_argument(
        "--no-pr",
        action="store_true",
        help="Skip the PR assembly + publish step (still writes the run directory).",
    )
    p_run.add_argument(
        "--runs-dir",
        default="./spinwright-runs",
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
    p_optimize.add_argument(
        "workspace",
        help="Path to a workspace built by `spinwright prep` or `spinwright extract`.",
    )
    p_optimize.add_argument(
        "--extraction",
        required=True,
        metavar="NAME",
        help="Extraction name; see `spinwright measure --help` for accepted forms.",
    )
    p_optimize.add_argument("--config", default=None, help="Path to spinwright.toml.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prep":
        from spinwright.cli import prep

        return prep.run(args)
    if args.command == "candidates":
        from spinwright.cli import candidates

        return candidates.run(args)
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
