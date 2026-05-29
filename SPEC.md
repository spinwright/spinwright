# Spinwright — SPEC

> A specialized agent that uses an LLM to find performance bottlenecks in Python code and propose targeted improvements as pull requests.

## 1. Overview

Spinwright is a small, purpose-built coding agent. Given a Python repository, it selects (or is given) a pytest test, extracts a clean repeatable harness around the code under test, measures baseline performance, profiles it with multiple tools, identifies bottlenecks through depth-first descent of hot call stacks, proposes and applies code improvements, verifies correctness against the full test suite, and opens a pull request bundling the accepted improvements.

The system is implemented as a Python program that orchestrates a sequence of Claude API calls. Claude is given a small, curated tool surface focused on performance analysis. The orchestrator (not the LLM) controls the outer loop, budgets, and termination.

The initial target is the static-frame repository, but Spinwright is designed to work on any pure-Python repository with a discoverable test suite. Tests written in either `unittest` style or plain `pytest` style are both supported; `pytest` is used purely as the discovery and execution runner, since it natively collects and runs `unittest.TestCase` subclasses.

**Platform:** Linux and macOS only. Windows is out of scope.

## 2. Goals and Non-Goals

### 2.1 Goals

- Reproduce, in automated form, the manual performance-analysis workflow the author has refined over years.
- Find real performance improvements (≥ 20% reduction in Callgrind instruction count for the targeted test) with zero correctness regressions in the full test suite.
- Produce reviewable, well-documented PRs that a human maintainer can evaluate quickly.
- Build up a persistent corpus of extracted tests in a configurable in-repo directory (see §9) so subsequent runs accelerate.
- Be runnable both locally and as a GitHub Action.

### 2.2 Non-Goals (v1)

- Multi-language support; Python only.
- Windows support.
- Optimizations requiring new C extensions, Cython, or Rust modules. Pure-Python edits only. Swapping a pure-Python implementation for an already-available stdlib/NumPy/SciPy equivalent is allowed.
- Algorithmic refactors that change public APIs.
- Memory profiling and memory-focused optimization (memray is not in scope).
- Cross-version Python testing within a single run.
- GPU-related work.
- Multi-test parallel optimization (single agent run at a time).
- Tests using pytest fixtures (other than `tmp_path`), parametrization, Hypothesis, or unittest `setUp`/`tearDown` methods with non-trivial state (see §5.2).

## 3. High-Level Workflow

```
1. Clone target repo to an isolated workspace.
2. Install repo + dev dependencies in a fresh venv.
3. Select candidate test(s):
     a. If user provided a test nodeid, use it.
     b. Otherwise, run full suite with --durations=0; pick slow tests;
        pass names+docstrings to LLM to rank likely-hot candidates.
4. For the chosen test:
     a. Check {corpus_dir} for an existing extraction. If
        present and current (matches commit SHA of source), reuse it.
     b. Otherwise, extract the test into setup() / run() / verify()
        functions in a new module under {corpus_dir}.
        Reject the test if it uses fixtures, parametrize, Hypothesis,
        non-trivial unittest setUp/tearDown, or anything else that
        prevents clean extraction.
     c. Commit the new extraction (and any notes) on the working branch.
5. Establish baselines: Callgrind instruction count (single execution),
   wall-clock time (best of K trials).
6. Enter the agent loop (§7):
     - Profile, descend, propose, apply, measure, accept/reject.
     - Continue until budget exhausted or no promising work remains.
7. If any patches were accepted:
     a. Run the full pytest suite. On failure, bisect and drop the
        offending patch(es); re-run.
     b. Generate PR body with measurements and reasoning.
     c. Push branch and open PR (GitHub Action mode) or print patch
        and report (local mode).
8. Otherwise, exit with a "no improvements found" report.
```

## 4. Architecture

Spinwright is a Python package with three primary layers:

```
spinwright/
├── orchestrator/         # Outer control loop; budgets; PR assembly
├── tools/                # Functions exposed to the LLM as tools
├── llm/                  # Anthropic API client; model selection;
│                         #   prompt templates; tool-call dispatch
├── extraction/           # Test discovery, eligibility checks, harness
│                         #   generation
├── measurement/          # Callgrind + timeit wrappers; result types
├── profiling/            # cProfile, pyinstrument, line_profiler,
│                         #   callgrind-graph adapters
├── repo/                 # Workspace setup; git operations; venv mgmt
└── cli/                  # Entrypoint(s) and config
```

### 4.1 Control Flow

- The **orchestrator** is plain Python. It owns: workspace, budgets, accepted-patch list, baseline measurements, the outer loop, and PR construction.
- The **LLM** is invoked through a small set of decision points, each a self-contained conversation with a defined tool set. Decision points include: candidate ranking, test extraction, bottleneck identification, patch proposal.
- **Tool calls** dispatch into `spinwright.tools`, which wraps the measurement/profiling/repo subsystems. Tools return structured results that are summarized by Claude Haiku 4.5 before being shown to the reasoning model when summarization is needed (e.g., large profile outputs).
- The orchestrator can interrupt and end any LLM conversation if budgets are exceeded.

### 4.2 Model Selection

- Reasoning steps (extraction planning, bottleneck identification, patch proposal): **Claude Opus 4.7**.
- Tool-output summarization (compressing long profiler output before it enters the reasoning model's context): **Claude Haiku 4.5**.
- Both invoked via the Anthropic API. Model strings configurable.

### 4.3 Implementation Language

Spinwright is implemented in **Python 3.11+**.

This is a deliberate choice driven by where Spinwright's load-bearing complexity lives:

- The eligibility checker and test extractor operate on Python source code at the AST level. Python's stdlib `ast` module is the canonical tool; alternatives (e.g., tree-sitter-python from Rust) are one abstraction removed and lack semantic awareness of import resolution and decorator binding.
- `line_profiler`, `cProfile`, and `pyinstrument` profile Python functions passed by reference, most naturally driven from the same Python process that imports the target package. From a non-Python orchestrator, each measurement would require writing a Python driver script to a temp file and parsing its stdout — workable but adds a layer.
- Patch validation includes `compile()`-ing the modified source to catch syntax errors before measurement; this is trivial in Python and requires bindings elsewhere.
- The agent edits Python code; having Python's parser/compiler available natively is a substantive ergonomic advantage.

Pure orchestration (git, valgrind subprocess, GitHub API) is straightforward in either language, so it doesn't drive the choice.

If Spinwright later needs to support non-Python target repos, the language-specific extractor and measurement subsystems would become pluggable; the orchestrator could be reimplemented in Rust at that point if motivated. That is out of scope for v1.

## 5. Test Selection and Extraction

### 5.1 Candidate Selection

If the user supplies a test nodeid (either pytest-style `tests/test_foo.py::test_bar` or unittest-style `tests/test_foo.py::TestThing::test_bar`), use it directly; eligibility is still checked (§5.2).

Otherwise:
1. Run the full suite once with `pytest --durations=0 -p no:randomly --tb=no -q` and parse the durations table. A single invocation enumerates every test that ran AND records each test's call-phase wall time, so we don't need a separate `--collect-only` pass. Pytest natively collects both pytest functions and `unittest.TestCase` subclasses, so this works for either style. (Tests that fail at collection or are skipped don't appear; a user can still target them explicitly via `--test`.)
2. Filter to tests with call-phase duration ≥ T_slow (default: 100 ms).
3. Filter to tests that pass the eligibility check (§5.2).
4. Pass remaining candidates (name, docstring, source) to the LLM with a ranking prompt: "Which of these tests is most likely to exercise hot code paths that have meaningful optimization headroom?" Return the top K (default: 5).
5. The orchestrator iterates through ranked candidates in order until one yields an accepted patch or all are exhausted.

### 5.2 Eligibility (Skip Conditions)

A test is **ineligible** for extraction if it:

- **(pytest-style)** Uses any pytest fixture in its function signature, other than `tmp_path` (which we handle by materializing a `tempfile.mkdtemp()` in `setup()`).
- **(pytest-style)** Is decorated with `@pytest.mark.parametrize`, `@pytest.mark.skip*`, or any marker that changes how the test is invoked.
- **(unittest-style)** Is a method on a `unittest.TestCase` subclass whose `setUp`/`setUpClass`/`tearDown`/`tearDownClass` methods are non-trivial (anything beyond `pass`, a simple attribute assignment, or a `super().setUp()` call that we can duplicate into our `setup()`).
- Imports from `hypothesis` or uses `@given`. (Import-level rule.)
- Imports symbols from a sibling `conftest.py` (v1 limitation; v1.5 may resolve and inline these). (Import-level rule.)
- *Calls* into network, filesystem (outside `tempfile`), or subprocess operations *in the test body or its helpers*. (Call-site rule — module-level `import subprocess` alone does not disqualify the test, only an actual `subprocess.run(...)` in the test path does. Files commonly bundle utilities used by a subset of tests; rejecting the whole module would shrink the candidate pool too aggressively.)
- Uses unseeded randomness (`random.*` without a prior `random.seed`, or `np.random.*` without a prior `np.random.seed`).

Eligibility is checked by a combination of AST inspection (deterministic, runs first) and an LLM check (handles cases like unseeded randomness inside imported helper functions, or unittest setup methods whose triviality the AST check can't confidently judge).

If a candidate is ineligible, log the reason and move to the next candidate.

### 5.3 Extraction

For an eligible test, the agent (LLM-driven, using `tools.read_source`, `tools.write_file`, and `tools.run_python`) produces a new module under the **configured corpus directory** (see §9). The default is `spinwright/` at the repo root — the contents are uniquely spinwright's (setup/run/verify harnesses named after their source nodeid) so a plain top-level name reads honestly. Repos that prefer the corpus nested under their own test tree can override (e.g., `static_frame/test/spinwright/`). Throughout this SPEC, `{corpus_dir}` refers to that configured path.

```
{corpus_dir}/<sanitized_test_id>.py
```

The module exposes:

```python
def setup() -> dict:
    """Build inputs and any state required by run(). Called once per
    measurement session. May seed RNGs. Must be deterministic."""

def run(state: dict) -> None:
    """The hot path under measurement. Called N times. Must not
    mutate `state` in a way that affects subsequent calls. No
    assertions here."""

def verify(state: dict) -> None:
    """Correctness check. Called once after the measurement loop.
    Contains the original test's assertions, adapted to read from
    state and/or re-invoke the operation as needed."""
```

Notes (`NOTES.md` adjacent to the extracted file) record:
- Original test nodeid and source commit SHA.
- Extraction date and Spinwright version.
- Any setup code duplicated from the source.
- Any deviations from the original test (e.g., "removed assertion on stdout").

The extraction is committed to the working branch immediately (`git add` + `git commit`) so subsequent runs see it. The commit message format: `spinwright: extract {test_id}`.

### 5.4 Reusing Extractions

On a subsequent run targeting the same test, if `{corpus_dir}/<test_id>.py` exists and the source-commit-SHA recorded in `NOTES.md` matches the current HEAD's view of the test's source file (i.e., the test hasn't changed), reuse the extraction. Otherwise, re-extract and overwrite, committing a new extraction.

## 6. Measurement

### 6.1 Two Measurements

For each measurement of an extracted test:

- **Callgrind instruction count (Linux only).** `valgrind --tool=callgrind --instr-atstart=no` invokes a small driver that does `setup()`, toggles instrumentation on, runs `run(state)` *N* times where *N* is auto-scaled (see §6.2), toggles off, then calls `verify(state)`. The instruction count is read from the callgrind output file and divided by *N* to report per-call instructions. **This is the primary metric on Linux.** Deterministic across runs; insensitive to system load.
- **Wall-clock time.** `timeit.Timer.autorange` picks an iteration count *N* such that one repeat takes ≥ 0.2 s, then runs *K* repeats (default *K*=5). Reports best, median, stddev. The sanity check on Linux; the **primary metric on macOS** (see §6.4).

Both measurements use the same extracted module; only the driver differs. Each measurement runs in a **fresh subprocess** of the target venv's Python — no in-process reuse across measurements. This isolates module-cache state, GC behavior, and import order from prior runs, and contains crashes in patched code so the orchestrator can recover.

### 6.2 Auto-Scaling

Single-shot Callgrind on a fast `run()` (sub-millisecond) buries the operation in Python interpreter and dispatch overhead, producing noisy per-call instruction counts. The driver therefore runs `run(state)` *N* times inside the instrumented region with *N* chosen so total instructions ≥ 1×10⁹ (configurable as `measurement.autoscale_min_instructions`). *N* is estimated from a cheap pre-measurement wall-time probe and then divided out of the reported per-call count. This mirrors `timeit.Timer.autorange`'s amortization logic.

### 6.3 Improvement Threshold

A patch is **accepted** if the Callgrind per-call instruction count for `run` decreases by at least 20% relative to the current baseline, with the full test suite still passing.

Within a single Spinwright run, multiple accepted patches stack; each new patch is measured against the cumulative baseline (i.e., with prior accepted patches applied). The final PR-level improvement reported is from the original baseline to the final cumulative state.

### 6.4 macOS Fallback

Valgrind has no working Apple Silicon port and recent x86_64 Darwin support is broken in practice. On macOS, Spinwright **skips Callgrind entirely** and reports only wallclock numbers. This means the macOS path is local-dev feedback, not the authoritative gate: the canonical gate (≥ 20% Callgrind instruction reduction) is only enforced on Linux, where Callgrind works. Run the agent loop on Linux for shippable results; use macOS for the extraction step and for ad-hoc wallclock sanity checks.

### 6.5 Verification of Correctness

After `run` executes under measurement, `verify(state)` is called. If `verify` raises, the measurement is considered invalid and the patch is rejected — regardless of instruction count or wall time.

### 6.6 Baseline Caching

Measurements are cached under `{corpus_dir}/.spinwright/baselines.json` keyed by `(extraction_blob_sha, target_source_sha)`. If neither has changed since a prior run, we reuse the cached baseline instead of re-measuring (Callgrind in particular is expensive — 10–100× slowdown on the instrumented region).

## 7. The Agent Loop

The agent loop is a depth-first descent through hot call stacks.

### 7.1 Loop State

The orchestrator maintains:
- `baseline`: current cumulative instruction count and wall time.
- `accepted_patches`: list of accepted diffs with metadata (target function, bottleneck description, before/after measurements, reasoning).
- `explored`: set of `(function_qualname, line_number)` pairs the agent has already examined, to avoid loops.
- `budget`: remaining token budget for the run.

### 7.2 One Iteration

Each iteration the LLM is given:
- The extracted test's source.
- The current baseline measurements.
- The list of accepted patches so far (as a summary).
- The current call-stack focus (initially: top of cProfile output; thereafter: the function currently being descended into).
- Available tools (§8).

The LLM is instructed to:
1. If at the top of the descent: run cProfile and pyinstrument; identify the top hot function in the target package's own code (filtering stdlib, NumPy, test harness). This becomes the current focus.
2. Run `line_profiler` on the current focus function, having added it to the profiler's target list.
3. Identify the hottest line(s).
4. If the hottest line is a call into another function in the target package: that callee becomes the new focus (descend; push current focus onto a stack).
5. If the hottest line is plain Python code in the current function with apparent improvement potential: propose a patch.
6. If neither (e.g., hottest line is a NumPy or stdlib call we can't edit): pop the stack; consider the next-hottest line in the parent focus. If the stack is empty: this branch is exhausted; restart from cProfile to find the next hot function not in `explored`.

### 7.3 Patch Proposal and Measurement

When the LLM proposes a patch:
1. The orchestrator applies the patch via `tools.apply_patch` (unified diff).
2. The orchestrator re-runs the extracted test under Callgrind and timeit.
3. If `verify` fails or instruction count increased: revert; the LLM is informed and may try a different approach on the same line/function.
4. If instruction count decreased by ≥ 20%: **accept**. Commit on the branch. Update baseline. The orchestrator decides whether to continue (budget remaining and promising bottlenecks left) or proceed to PR assembly.
5. If instruction count decreased by < 20%: **soft accept** in a staging area. Continue descending; if cumulative soft-accepted patches reach ≥ 20%, they all promote to accepted. If the run ends with soft-accepted patches summing < 20%, they are discarded.

### 7.4 Stopping Conditions

The agent loop ends when any of these is true:
- Token budget for the run is exhausted.
- The LLM reports no remaining promising bottlenecks (and the orchestrator confirms via `explored` coverage of the top P% of cProfile output).
- A hard iteration cap (default: 30) is hit.
- An infrastructure error occurs (e.g., the patched code crashes Python in a way that breaks the measurement harness; the orchestrator reverts the offending patch and stops).

### 7.5 Suite Regression Check

After the loop ends, if there is at least one accepted patch:
1. Run the full pytest suite.
2. If green: proceed to PR assembly.
3. If red: identify failing tests. Bisect the accepted-patch list (binary search) to find the offending patch(es); revert them. Re-run the suite.
4. Repeat until green or all patches are reverted. If all patches are reverted, exit with a "regression" report (no PR).

## 8. Tool Surface

These are the tools exposed to the LLM. All return structured data; large outputs are summarized by Haiku before being shown to the reasoning model.

### 8.1 Discovery and Source

- `list_tests(pattern: str | None) -> list[TestNodeId]`
  Lists pytest tests, optionally filtered by name pattern.

- `get_test_source(nodeid: TestNodeId) -> {path, source, lineno, decorators}`
  Returns the source of a test and any decorators on it.

- `read_source(qualname: str) -> {path, source, lineno}`
  Resolves a Python qualname (e.g., `static_frame.core.index.Index._loc_to_iloc`) to its source code and location.

- `list_callers(qualname: str) -> list[qualname]` — *optional v1.5*
  Uses static analysis to list possible callers of a function within the target package.

### 8.2 Extraction Helpers

- `check_eligibility(nodeid: TestNodeId) -> {eligible: bool, reasons: list[str]}`
  Runs the AST-based eligibility check.

- `write_extraction(test_id: str, module_source: str, notes: str) -> path`
  Writes the extracted module and `NOTES.md` to `{corpus_dir}/` (default `spinwright/`), commits on the working branch.

### 8.3 Measurement

- `measure_callgrind(extraction_path: str) -> {instructions: int, output_path: str}`
  Runs the extraction under Callgrind. Verifies correctness via `verify()`.

- `measure_walltime(extraction_path: str, iterations: int | None, repeats: int) -> {best, median, stddev, iterations_used}`
  Runs the extraction under timeit.

### 8.4 Profiling

- `profile_cprofile(extraction_path: str, sort: str = "cumulative", top: int = 30) -> ProfileSummary`
- `profile_pyinstrument(extraction_path: str, interval: float = 0.0001) -> ProfileSummary`
- `profile_line(extraction_path: str, target_qualnames: list[str]) -> LineProfileSummary`
- `profile_callgrind_graph(extraction_path: str, threshold_node: float, threshold_edge: float) -> {dot_path, top_nodes}`

Each profile tool returns both a path to the raw output and a structured summary suitable for LLM consumption.

### 8.5 Code Modification

- `apply_patch(diff: str) -> {success: bool, message: str}`
  Applies a unified diff. Validates syntax (Python compiles) before returning success.

- `revert_patch(patch_id: str) -> None`
  Reverts a previously applied patch.

- `git_diff() -> str`
  Returns the current diff against the branch base.

### 8.6 Test Running

- `run_extracted_test(extraction_path: str) -> {passed: bool, error: str | None}`
  Calls `setup()`, then `run(state)` once, then `verify(state)`. Used by the LLM to sanity-check a change before requesting a full measurement.

- `run_pytest(scope: "module" | "full", path: str | None) -> {passed, failed, errors, duration}`
  Runs the full test suite or a specific module.

### 8.7 Repo and PR

- `git_branch(name: str) -> None`
- `git_commit(message: str) -> sha`
- `git_push(remote: str = "origin") -> None`
- `open_pr(title: str, body: str, base: str = "main") -> pr_url`
  GitHub Action mode only; local mode prints the PR title/body and patch series.

## 9. Configuration

Spinwright is configured via a TOML file (`spinwright.toml` in the target repo or via `--config`) and CLI flags. Key settings:

```toml
[target]
repo_url = "https://github.com/static-frame/static-frame"
ref = "master"

[corpus]
# Directory inside the target repo where extracted tests are stored
# and committed. For static-frame: "static_frame/test/spinwright".
dir = "spinwright"

[test_selection]
slow_threshold_seconds = 0.1
top_k_candidates = 5
explicit_nodeid = ""  # if set, skip discovery

[measurement]
improvement_threshold = 0.20
walltime_repeats = 5
callgrind_path = "valgrind"

[budget]
tokens_per_run = 2_000_000
max_iterations = 30
max_wall_clock_minutes = 60

[models]
reasoning = "claude-opus-4-7"
summarization = "claude-haiku-4-5-20251001"

[pr]
mode = "local"  # or "github_action"
base_branch = "master"
branch_prefix = "spinwright/"
```

## 10. PR Format

PR title:
```
perf({module}): {short description} (−{X}% instructions on {test_name})
```

PR body template:

```markdown
## Summary

Spinwright identified and applied {N} optimization(s) to `{module}`,
reducing Callgrind instruction count by **{X}%** on the extracted
test `{test_id}`.

## Measurements

| Metric                  | Baseline     | After       | Δ       |
|-------------------------|--------------|-------------|---------|
| Callgrind instructions  | {baseline_i} | {final_i}   | −{X}%   |
| Wall time (median, ms)  | {baseline_w} | {final_w}   | −{Y}%   |
| Wall time (stddev, ms)  | {baseline_s} | {final_s}   |         |

Wall time measured over {K} repeats of {N} iterations each.

## Test

The extracted test harness is at `{corpus_dir}/{test_id}.py`,
derived from `{original_test_nodeid}` at commit `{sha}`.

The full pytest suite passes ({pass_count} tests).

## Bottlenecks and Changes

### {bottleneck_1_title}

{description of what was slow, where, and why}

**Change:** {brief description}

{diff excerpt}

**Local impact:** {instruction delta on this change alone}

---

{repeat per accepted patch}

## Notes

Generated by Spinwright v{version} using {reasoning_model}.
Run ID: {run_id}.
```

## 11. CLI

```
spinwright prep    <repo_url_or_path>                          # clone + venv + install, prints workspace path
spinwright extract <repo_or_workspace> --test <nodeid>         # LLM-driven extraction, auto-detects workspace reuse
spinwright measure <workspace> --extraction {corpus_dir}/<id>.py
spinwright run     <repo_url> [options]                        # full agent loop (M3+)
spinwright report  <run_id>                                    # M6
```

Flags:
- `--config PATH`
- `--test NODEID`           (skip discovery; target this test)
- `--dry-run`               (do everything but the PR open)
- `--verbose` / `--quiet`

`extract` auto-detects whether its first arg is a path to an existing prep'd workspace (has `.venv/bin/python` and `repo/.git`) and reuses it in place; otherwise it preps a fresh one. Workspaces from `prep` and `extract` are always kept on disk (the user just paid for an LLM extraction); cleanup is on the caller.

## 12. GitHub Action

A reusable workflow at `.github/workflows/spinwright.yml` (template provided):

```yaml
on:
  workflow_dispatch:
    inputs:
      test:
        description: "Specific test nodeid (optional)"
        required: false
jobs:
  spinwright:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get install -y valgrind
      - uses: actions/setup-python@v5
      - run: pip install spinwright
      - env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: spinwright run . --test "${{ inputs.test }}" --config spinwright.toml
```

Required secrets: `ANTHROPIC_API_KEY` for the LLM, `GITHUB_TOKEN` (or a PAT with PR scope) for opening PRs.

## 13. Persistent Artifacts in the Target Repo

Spinwright commits and maintains, at the configured `{corpus_dir}`:

```
{corpus_dir}/                     # e.g., static_frame/test/spinwright
├── README.md                     # Generated; explains what this dir is
├── <test_id_1>.py                # Extracted harness
├── <test_id_1>.NOTES.md          # Metadata, deviations, history
├── <test_id_2>.py
├── <test_id_2>.NOTES.md
└── .spinwright/
    ├── run_log.jsonl             # Append-only log of runs (test, result)
    └── bottleneck_history.json   # Functions previously optimized, to
                                  #   avoid re-attempting unchanged code
```

`bottleneck_history.json` lets later runs skip already-optimized functions whose source hasn't changed.

The extracted modules are valid Python files placed inside the target repo's tree but are **named so as not to be auto-collected by pytest** (e.g., they do not begin with `test_`). They are only invoked by Spinwright's own drivers. The corpus directory's `__init__.py` (if needed for import) is generated by Spinwright on first write.

## 14. Logging and Observability

Each run produces:
- A structured JSONL log of every LLM turn, tool call, and measurement.
- A run-summary JSON with: test selected, baseline, accepted patches, final delta, token usage by model, total wall time.
- The raw outputs of every profile tool, retained in the run directory.

Local mode: written to `./spinwright-runs/<run_id>/`.
GitHub Action mode: uploaded as a workflow artifact.

## 15. Failure Modes and Recovery

Spinwright is expected to handle:

- **Repo install failure:** abort with a clear error; no PR.
- **No eligible test candidates:** exit cleanly with "nothing to do" report.
- **Extraction fails repeatedly:** mark the candidate as ineligible (in `bottleneck_history.json`) and move on.
- **Measurement infrastructure failure** (Callgrind segfault, OOM): retry once; if it persists, skip the candidate.
- **Suite regression after the loop:** bisect-and-revert (§7.5).
- **Budget exhaustion mid-loop:** wrap up cleanly — if any patches are firmly accepted, still attempt the PR; if only soft-accepted, discard.
- **Network failure on PR open:** log the patch series locally so the user can manually open the PR.

## 16. Open Questions for Implementation

(Items to resolve during build; not blockers for the SPEC.)

1. The "filter out NumPy / stdlib" rule for hot-function selection — implement as a module-path prefix check (target package only) or also allow descending into pure-Python dependencies?
2. Threshold for `slow_threshold_seconds` may need per-repo tuning; consider adaptive selection (e.g., top P% by duration rather than absolute threshold).
3. Whether to allow the agent to read `git log` / `git blame` on candidate hot functions, which could inform reasoning ("this was last touched 5 years ago and pre-dates a NumPy API that would now be faster").
4. Conftest discovery for extracted tests when the test's module imports from a sibling conftest — default for v1 is "ineligible", configurable via `eligibility.allow_pure_conftest_imports`.

## 17. Milestones

1. **M1 — Extraction & measurement. ✅** Repo setup, eligibility checker, extraction (LLM-driven), wallclock measurement. End-to-end CLI (`prep`, `extract`, `measure`) working against any pip-installable Python repo.
2. **M2 — Single-shot optimization. ✅ (mostly)** Callgrind (Linux, via two-run subtraction since the `CALLGRIND_*` macros need inline assembly), cProfile profiling, single-iteration optimization loop with profile→propose→measure→accept-or-revert via `spinwright optimize`. The orchestrator currently gates on median wallclock on every platform; switching it to the Callgrind gate on Linux is a small follow-up (already-built `measurement.callgrind` is wired into the `measure` CLI but not yet into the `optimize` decision path). pyinstrument and line_profiler tools deferred until needed — cProfile is the LLM's signal for now.
3. **M3 — Agent loop + regression check. ✅ (BFS, not DFS)** Full agent loop with `explored` tracking, focus-hinted optimization, linear-revert-with-drop-all fallback on regression, `spinwright run` end-to-end CLI. (Soft-accept is out — see Appendix A, Mod 6.) Depth-first descent into callees within a function is deferred until `line_profiler` lands — the current loop is BFS over top-cumtime functions in the target package, which is the practical equivalent without line-level profiling. Multi-patch bisect refinement (when no single revert restores green) is also deferred; the fallback is conservative drop-all.
4. **M4 — PR assembly & local CLI. ✅** PR title + body rendering (SPEC §10 template), per-run artifact directory under `./spinwright-runs/<run_id>/` with `PR.md` and `run_summary.json`, local-mode that always writes PR.md, github_action-mode that pushes the branch and calls `gh pr create --body-file` (falls back to local with a clear reason when `gh` is missing or `git push` fails). Wired into `spinwright run` with `--no-pr` and `--runs-dir` flags.
5. **M5 — GitHub Action.** Workflow template, secrets handling, artifact upload. Persistent `{corpus_dir}/` (default `spinwright/`) and `bottleneck_history.json`.
6. **M6 — Hardening.** Failure-mode handling, budget enforcement, structured logging, run-summary reporting.

## Appendix A. Modifications adopted from M1 planning

Recorded during the M1 plan review (see `~/.claude/plans/lets-implement-spinwright-see-splendid-llama.md`). Items 1–3 and 7 are reflected inline above; the rest are listed here because their corresponding subsystems aren't built yet, but the design they imply is fixed and should be honored when the milestone arrives.

| # | Modification | Status in v1 |
|---|--------------|--------------|
| 1 | **macOS measurement fallback.** Skip Callgrind entirely on Darwin; wallclock only. Linux remains the canonical metric. | §6.4 |
| 2 | **Callgrind auto-scaling.** Pick *N* so total instructions ≥ 1×10⁹; report per-call as total/*N*. | §6.2 |
| 3 | **Measurements run as fresh subprocesses.** No in-process reuse. | §6.1 |
| 4 | **Edit tool replaces unified-diff `apply_patch`.** Tool shape: `edit_file(path, old_string, new_string)` with uniqueness check; orchestrator reconstructs unified diffs for PR bodies. | Implemented in M1's `tools.edit_file`; relevant from M2 onward. |
| 5 | **Model tiering.** Sonnet 4.6 for classification (eligibility second-pass, candidate ranking); Opus 4.7 for reasoning (extraction, bottleneck ID, patch proposal); Haiku 4.5 for summarization. | Config knobs live in `config.models.{reasoning,classification,summarization}`; extraction uses reasoning today. |
| 6 | **Drop "soft accept".** Only patches with ≥ 20% individual Callgrind improvement are accepted. Removes the §7.3.5 promotion rule from the original draft. | Codified now; relevant from M3. |
| 7 | **Single-run pytest discovery.** One invocation with `--durations=0` enumerates and times every test that ran. (The original draft had a separate `--collect-only` pass; that pass only finds tests but doesn't time them, so it can't be merged with the durations run — instead it's redundant and removed entirely. Tests that fail at collection or are skipped won't appear in discovery, which is fine because the user can target them explicitly via `--test`.) | §5.1 |
| 8 | **Baseline caching.** `(extraction_blob_sha, target_source_sha) → baseline` under `{corpus_dir}/.spinwright/baselines.json`. | §6.6; relevant from M2. |
| 9 | **Linear-revert before bisect.** Try reverting each accepted patch individually first; fall back to bisect only when no single revert restores green. | §17 (M3 description); relevant from M3. |
| 10 | **Reframe iteration budget.** Drop ambiguous `max_iterations=30`; use `max_patches_proposed` and `max_descents_per_focus`. | Config knobs live; relevant from M3. |
| 11 | **Sandbox: subprocess + temp dir.** Working tree under `tempfile.mkdtemp()`; `edit_file` validates the path is inside it; every measurement is a subprocess. No Docker dep for v1. | Implemented in M1's `repo.workspace`, `tools.edit`, `measurement.runner`. |
| 12 | **Conftest carve-out flag.** `eligibility.allow_pure_conftest_imports`; default False. | §5.2; configurable. |

### Additional revisions discovered during M1 implementation

- **Eligibility narrowing (§5.2).** Network/subprocess/filesystem ineligibility is **call-site only**, not import-level. The original wording would over-reject test modules that import `subprocess` for one test but use only pure-Python paths in the test we're targeting. Only `hypothesis` and `conftest` are import-level rules per spec.
- **Sanity check after extraction.** The LLM-driven extraction is post-validated by running `setup → run → verify` once inside the target venv before NOTES.md is written or the commit lands. Broken extractions are not committed; the failure is reported back to the caller for retry or escalation.
