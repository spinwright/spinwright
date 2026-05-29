from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from spinwright.repo.workspace import Workspace


def python_executable(ws: Workspace) -> Path:
    return ws.venv_dir / "bin" / "python"


def pip_executable(ws: Workspace) -> Path:
    return ws.venv_dir / "bin" / "pip"


def create(ws: Workspace) -> None:
    subprocess.run(
        [sys.executable, "-m", "venv", str(ws.venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(python_executable(ws)), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        capture_output=True,
        text=True,
    )


def install_target(
    ws: Workspace,
    extras: tuple[str, ...] = (),
    extra_packages: tuple[str, ...] = ("pytest",),
) -> None:
    target = "." if not extras else f".[{','.join(extras)}]"
    subprocess.run(
        [str(pip_executable(ws)), "install", "-e", target],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ws.repo_dir),
    )
    if extra_packages:
        subprocess.run(
            [str(pip_executable(ws)), "install", *extra_packages],
            check=True,
            capture_output=True,
            text=True,
        )
