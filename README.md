# spinwright

LLM-driven Python performance-optimization agent. Given a Python repo with a
test suite, spinwright picks a slow test, extracts it into a measurement
harness, drives an LLM agent that profiles + edits + measures, and opens a PR
with the surviving optimizations.

## Install

Spinwright targets Python 3.11+. Until it lands on PyPI:

```sh
pip install spinwright
```

On Linux, install valgrind too so the optimization gate uses deterministic
Callgrind instruction counts (the macOS path falls back to wallclock):

```sh
sudo apt-get install -y valgrind   # or: brew, dnf, pacman, etc.
```

Set the API key for whatever LLM provider you'll use:

```sh
export ANTHROPIC_API_KEY=sk-ant-...   # for anthropic/claude-*
export OPENAI_API_KEY=sk-...          # for openai/gpt-* and openai/o[1-9]-*
export OLLAMA_API_KEY=...             # for ollama/* against Ollama Cloud
```

## How it fits together

Spinwright is six small commands that pipeline into each other. Each owns one
thing; they share a *workspace* (a temp directory holding a clone of the
target repo plus its own venv).

```
prep ──► candidates ──► extract ──► measure
                            │           │
                            ▼           ▼
                          run ──► optimize
                            │
                            ▼
                          PR.md + git commits
```

| Command | What it does | LLM? |
|---|---|---|
| `prep` | Clone target repo, build a venv, install it | no |
| `candidates` | Run `pytest --durations=0`, AST-filter for eligible slow tests | no |
| `extract` | LLM rewrites one test as `setup()/run(state)/verify(state)` | yes |
| `measure` | Wall-clock (timeit autorange) + Callgrind on Linux | no |
| `optimize` | One round: profile → propose a patch → measure → accept-or-revert | yes |
| `run` | Iterate `optimize` up to the budget, regression-check, write PR.md | yes |

## CLI

Each command takes the workspace path as a positional arg. `--help` on any
command prints the full flag set.

### `prep` — build a workspace

Clones the target repo into a temp dir (or a path you name), creates a fresh
venv, installs the target editable, plus any requirements file you point at.

```sh
spinwright prep <repo>                                  # URL or local path
spinwright prep <repo> --workspace ~/spinwright-sf      # stable path you'll reuse
spinwright prep <repo> -r requirements-test.txt         # repeatable
spinwright prep <repo> --extras dev,test                # optional [extras]
spinwright prep <repo> --ref v0.2                       # pin a git ref
```

Workspace path is printed on stdout (suitable for `WS=$(spinwright prep ...)`);
progress goes to stderr. Workspaces are always kept — you `rm -rf` them when
you're done.

### `candidates` — discover slow eligible tests

Runs the target's pytest suite once with `--durations=0`, filters by your
configured threshold (default 0.1s), then AST-filters for eligibility per
SPEC §5.2 (no fixtures except `tmp_path`, no parametrize/skip markers, no
hypothesis, no in-test network/subprocess/`open()`, seeded RNG).

```sh
spinwright candidates <workspace>
spinwright candidates <workspace> --test-path src/foo/tests   # subtree scope
spinwright candidates <workspace> --json                       # machine-readable
spinwright candidates <workspace> --nodeids                    # one nodeid/line
```

`--nodeids` is pipe-friendly: `spinwright candidates <ws> --nodeids | head -1`
gives you the slowest eligible nodeid in one line.

### `extract` — LLM-driven test extraction

The LLM rewrites the test into the measurement harness contract:
`setup() -> dict` builds the inputs, `run(state) -> None` is the hot path
(no assertions, no side-effects-on-state), `verify(state) -> None` checks
the result. Writes to `<repo>/<corpus_dir>/<sanitized_id>.py` (default
`<corpus_dir>` is `spinwright/`) plus a sibling `.NOTES.md` capturing the
source nodeid, commit SHA, and timestamp. Commits both on the working branch.

```sh
spinwright extract <workspace> --test 'tests/test_x.py::test_y'
```

Sanity-checks the extraction (one full `setup → run → verify`) before
committing. A broken extraction is reported and NOT committed.

### `measure` — benchmark an extracted harness

```sh
spinwright measure <workspace> --extraction <stem>      # default 5 repeats
spinwright measure <workspace> --extraction <stem> --repeats 10
spinwright measure <workspace> --extraction <stem> --json
```

`<stem>` is the sanitized test id (the bare filename without `.py`); pasting
the longer form `spinwright/<stem>.py` also works. Reports best/median/stddev
wallclock and (on Linux with valgrind) Callgrind instructions per call. Hints
the next step on success.

### `optimize` — one round of LLM optimization

Measures baseline → drives the LLM through one optimization conversation
(profile via cProfile, read source, propose ONE edit, sanity-check) →
remeasures → accepts if the configured `improvement_threshold` (default 20%)
is met and `verify()` still passes; otherwise reverts everything.

```sh
spinwright optimize <workspace> --extraction <stem>
spinwright optimize <workspace> --extraction <stem> --model openai/gpt-4o
```

On accept: commits to the working branch. On reject: prints the rejection
reason + a truncated diff of what the LLM tried.

### `run` — full agent loop

Iterates `optimize` up to `budget.max_patches_proposed` times,
tracking explored functions to avoid loops. Then runs the target's pytest
suite as a regression check and drops any patch that breaks it
(linear-revert; falls back to drop-all if no single revert restores green).
Finally builds `PR.md` + `run_summary.json` under `./spinwright-runs/<run_id>/`.

```sh
spinwright run <workspace> --extraction <stem>
spinwright run <workspace> --extraction <stem> --test-path src/foo/tests
spinwright run <workspace> --extraction <stem> --skip-regression
spinwright run <workspace> --extraction <stem> --no-pr
```

`--test-path` constrains the regression check to a subtree of the test suite.
Use it to keep CI runtime sane when the full suite is large.

## Model selection

The `--model` flag (or `[models].model` in `spinwright.toml`) picks the LLM.
Three providers; explicit `provider/model_id` always works, bare names are
auto-routed via heuristics.

| Spec | Provider | Heuristic for bare names |
|---|---|---|
| `anthropic/claude-opus-4-7` | Anthropic API | bare `claude-*` |
| `openai/gpt-4o`, `openai/o4-mini` | OpenAI API | bare `gpt-*`, `o[1-9]-*` |
| `ollama/llama3.1:8b` | Ollama Cloud (default) or self-hosted | none — always prefix |

Each provider reads one env var:
- `ANTHROPIC_API_KEY` for Anthropic
- `OPENAI_API_KEY` for OpenAI
- `OLLAMA_API_KEY` for Ollama Cloud (the default). Set `OLLAMA_HOST` to a
  self-hosted endpoint (e.g. `http://localhost:11434`) to bypass the key
  check — local servers don't authenticate.

## Configuration

Most flags can be persisted in `spinwright.toml` (referenced via `--config`).
The full schema lives in [`SPEC.md`](./SPEC.md) §9; the useful subset:

```toml
[corpus]
dir = "spinwright"           # where extracted harnesses live in the target repo

[test_selection]
slow_threshold_seconds = 0.1

[measurement]
improvement_threshold = 0.20

[budget]
max_patches_proposed = 10
max_extraction_turns = 30

[models]
model = "claude-opus-4-7"    # or "openai/gpt-4o", "ollama/llama3.1:8b"
```

## End-to-end local recipe

```sh
WS=$(spinwright prep https://github.com/static-frame/static-frame \
       --workspace ~/sf-ws \
       -r requirements-test.txt)

NID=$(spinwright candidates "$WS" --test-path static_frame/test/unit --nodeids | head -1)
spinwright extract "$WS" --test "$NID"

STEM=$(basename "$NID" | tr ':' '_')   # rough; `extract` prints the exact stem
spinwright measure  "$WS" --extraction "$STEM"
spinwright run      "$WS" --extraction "$STEM" --test-path static_frame/test/unit
cat ./spinwright-runs/run_*/PR.md
```

## GitHub Action

Spinwright ships a **reusable workflow**. The
pipeline lives in spinwright; the target repo only needs a ~30-line caller
stub.

### Setup (one-time, in the target repo)

1. Add the API key as a repository secret:
   - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OLLAMA_API_KEY` — whichever
     matches the model you'll target.
2. Drop the caller stub from
   [`examples/github-action.yml`](./examples/github-action.yml) at
   `.github/workflows/spinwright.yml` in the target repo.
3. Replace the `OWNER` placeholder (twice — in `uses:` and in
   `spinwright_install_spec`) with the GitHub org/user hosting the spinwright
   fork you trust.
4. Pin the `@ref` (e.g. `@v0.2.0`) once you cut a release so workflow updates
   don't silently change CI behavior.

### Triggering

The workflow is `workflow_dispatch`-only (manual trigger from the Actions
tab). Useful inputs:

| Input | Default | Effect |
|---|---|---|
| `test` | blank | Specific pytest nodeid; blank ⇒ auto-pick slowest eligible |
| `test_path` | blank | Repo-relative subtree; scopes BOTH auto-select AND the regression check |
| `requirements` | blank | Repo-relative path to a requirements file, forwarded to `prep -r` |
| `extras` | blank | Comma-separated `[project.optional-dependencies]` extras |
| `model` | blank ⇒ config default | e.g. `openai/gpt-4o`, `ollama/llama3.1:8b` |

The workflow runs on `ubuntu-latest` with valgrind, picks up the right API
key based on the model spec, runs the pipeline, opens a PR via `gh pr create`
on success, and uploads the per-run artifacts (`PR.md` + `run_summary.json`)
either way.

### Caveats

- **Reusable-workflow secrets aren't conditionally required.** All three
  API-key secrets are declared `required: false`; if the wrong one is set
  for the chosen model, spinwright raises a clear `MissingAPIKeyError` early.
- **`gh pr create` runs from the checkout, not from spinwright.** The
  workflow fetches commits from spinwright's working branch into the
  actions/checkout-authenticated copy, then pushes from there. This keeps
  spinwright out of CI git-auth handling.
- **PyPI install spec.** Currently the caller stub passes
  `spinwright_install_spec: 'git+https://github.com/OWNER/spinwright.git@main'`.
  Once spinwright is on PyPI, delete that line — the default is just
  `spinwright`.
