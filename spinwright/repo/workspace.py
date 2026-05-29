from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path           # tmpdir root
    repo_dir: Path       # clone destination (root / "repo")
    venv_dir: Path       # root / ".venv"
    branch: str
    base_sha: str
    keep: bool


def _git(repo_dir: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        # subprocess.CalledProcessError's default str() drops stderr, which is
        # where git puts its actual error message. Re-raise so the user sees it.
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _resolve_source(source: str) -> str:
    p = Path(source).expanduser()
    if p.exists():
        return str(p.resolve())
    return source


def create(
    source: str,
    ref: str | None,
    branch_prefix: str,
    branch_suffix: str,
    keep: bool = False,
    root: Path | None = None,
) -> Workspace:
    """Build a workspace from a repo source.

    ``root`` controls where the workspace lives:
      - ``None`` (default): ``tempfile.mkdtemp(prefix="spinwright-")``
      - explicit ``Path``: that exact directory. Created if it doesn't exist;
        must be empty otherwise (we refuse to clobber).
    """
    if root is None:
        root = Path(tempfile.mkdtemp(prefix="spinwright-"))
    else:
        root = root.expanduser().resolve()
        if root.exists():
            if any(root.iterdir()):
                raise FileExistsError(
                    f"workspace path {root} exists and is not empty — "
                    "remove it first or pick a different path"
                )
        else:
            root.mkdir(parents=True)
    repo_dir = root / "repo"
    venv_dir = root / ".venv"

    src = _resolve_source(source)
    subprocess.run(
        ["git", "clone", src, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    if ref:
        _git(repo_dir, "checkout", ref)
    base_sha = _git(repo_dir, "rev-parse", "HEAD")

    branch = f"{branch_prefix}{branch_suffix}"
    _git(repo_dir, "checkout", "-b", branch)

    return Workspace(
        root=root,
        repo_dir=repo_dir,
        venv_dir=venv_dir,
        branch=branch,
        base_sha=base_sha,
        keep=keep,
    )


def commit(ws: Workspace, paths: list[Path], message: str) -> str:
    rels = [str(p.relative_to(ws.repo_dir)) for p in paths]
    _git(ws.repo_dir, "add", *rels)
    _git(ws.repo_dir, "-c", "user.email=spinwright@localhost", "-c", "user.name=spinwright",
         "commit", "-m", message)
    return _git(ws.repo_dir, "rev-parse", "HEAD")


def cleanup(ws: Workspace) -> None:
    if ws.keep:
        return
    shutil.rmtree(ws.root, ignore_errors=True)


def reuse(root: Path) -> Workspace:
    """Reconstruct a Workspace from an existing prep'd directory.

    Used by ``spinwright extract`` when the user passes a path to a workspace
    built by ``spinwright prep``. ``keep`` is forced to True — we never delete
    a workspace we didn't create. ``base_sha`` is the current HEAD (the SHA
    that future extractions and NOTES.md entries will reference).
    """
    repo_dir = root / "repo"
    venv_dir = root / ".venv"
    if not (repo_dir / ".git").exists():
        raise FileNotFoundError(f"{root} is not a spinwright workspace (no repo/.git)")
    if not (venv_dir / "bin" / "python").exists():
        raise FileNotFoundError(f"{root} has no .venv/bin/python — run `spinwright prep` first")
    branch = _git(repo_dir, "branch", "--show-current")
    base_sha = _git(repo_dir, "rev-parse", "HEAD")
    return Workspace(
        root=root,
        repo_dir=repo_dir,
        venv_dir=venv_dir,
        branch=branch,
        base_sha=base_sha,
        keep=True,
    )
