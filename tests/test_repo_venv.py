from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from spinwright.repo import venv, workspace


def _tiny_src_repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n'
        '[project]\nname="tiny"\nversion="0.0.1"\nrequires-python=">=3.10"\n'
    )
    (src / "tiny").mkdir()
    (src / "tiny" / "__init__.py").write_text("def add(a, b): return a + b\n")
    # An empty requirements file is still a valid `pip install -r` target
    # and is the cheapest way to exercise the code path without bringing in
    # an unrelated package.
    (src / "requirements-extra.txt").write_text("# empty on purpose\n")
    for cmd in (
        ["git", "init", "-b", "main"],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "add", "."],
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", "commit", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=src, check=True, capture_output=True)
    return src


def test_install_target_accepts_requirements_file(tmp_path: Path):
    src = _tiny_src_repo(tmp_path)
    ws = workspace.create(
        source=str(src), ref=None,
        branch_prefix="spinwright/", branch_suffix="venv-test",
        keep=True,
    )
    venv.create(ws)
    # Should succeed with an empty requirements file present in the repo.
    venv.install_target(
        ws,
        extras=(),
        extra_packages=(),
        requirements_files=("requirements-extra.txt",),
    )
    # The editable install still landed.
    out = subprocess.run(
        [str(venv.python_executable(ws)), "-c", "import tiny; print(tiny.add(1, 2))"],
        check=True, capture_output=True, text=True,
    )
    assert out.stdout.strip() == "3"
    workspace.cleanup(ws)


def test_install_target_rejects_missing_requirements_file(tmp_path: Path):
    src = _tiny_src_repo(tmp_path)
    ws = workspace.create(
        source=str(src), ref=None,
        branch_prefix="spinwright/", branch_suffix="venv-test",
        keep=True,
    )
    venv.create(ws)
    with pytest.raises(FileNotFoundError, match="requirements file"):
        venv.install_target(
            ws,
            extras=(),
            extra_packages=(),
            requirements_files=("requirements-nope.txt",),
        )
    workspace.cleanup(ws)
