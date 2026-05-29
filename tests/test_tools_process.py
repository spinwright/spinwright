from __future__ import annotations

import sys
from pathlib import Path

from spinwright.tools import process


PY = Path(sys.executable)


def test_run_python_captures_stdout(tmp_path: Path):
    res = process.run_python(PY, "print('hello')", cwd=tmp_path)
    assert res.returncode == 0
    assert res.stdout.strip() == "hello"
    assert res.timed_out is False


def test_run_python_reports_nonzero_exit(tmp_path: Path):
    res = process.run_python(PY, "import sys; sys.exit(3)", cwd=tmp_path)
    assert res.returncode == 3
    assert res.timed_out is False


def test_run_python_captures_stderr(tmp_path: Path):
    res = process.run_python(PY, "raise ValueError('boom')", cwd=tmp_path)
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "boom" in res.stderr


def test_run_python_times_out(tmp_path: Path):
    res = process.run_python(
        PY, "import time; time.sleep(5)", cwd=tmp_path, timeout_seconds=0.5
    )
    assert res.timed_out is True
    assert res.returncode == -1


def test_run_python_inherits_cwd(tmp_path: Path):
    res = process.run_python(PY, "import os; print(os.getcwd())", cwd=tmp_path)
    assert res.returncode == 0
    assert str(tmp_path.resolve()) in res.stdout
