from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from spinwright import config


def test_default_has_expected_models():
    cfg = config.default()
    assert cfg.models.reasoning == "claude-opus-4-7"
    assert cfg.models.classification == "claude-sonnet-4-6"
    assert cfg.models.summarization == "claude-haiku-4-5-20251001"


def test_default_modification_knobs_present():
    cfg = config.default()
    assert cfg.measurement.autoscale_min_instructions == 1_000_000_000
    assert cfg.budget.max_patches_proposed == 10
    assert cfg.budget.max_descents_per_focus == 4
    assert cfg.eligibility.allow_pure_conftest_imports is False


def test_from_dict_partial_keeps_defaults():
    cfg = config.from_dict({"target": {"repo_url": "https://example.com/x"}})
    assert cfg.target.repo_url == "https://example.com/x"
    # Empty default — workspace.create() then skips `git checkout` and leaves
    # the clone on the remote default branch (which may be main, master, etc.).
    assert cfg.target.ref == ""
    assert cfg.corpus.dir == "spinwright"


def test_from_dict_full_round_trips():
    data = {
        "target": {"repo_url": "https://example.com/x", "ref": "v1"},
        "corpus": {"dir": "my/corpus"},
        "test_selection": {
            "slow_threshold_seconds": 0.5,
            "top_k_candidates": 3,
            "explicit_nodeid": "tests/test_x.py::test_y",
        },
        "eligibility": {"allow_pure_conftest_imports": True},
        "measurement": {
            "improvement_threshold": 0.3,
            "walltime_repeats": 7,
            "callgrind_path": "/usr/local/bin/valgrind",
            "autoscale_min_instructions": 500_000_000,
        },
        "budget": {
            "tokens_per_run": 1_000_000,
            "max_patches_proposed": 5,
            "max_descents_per_focus": 2,
            "max_wall_clock_minutes": 30,
            "max_extraction_turns": 15,
        },
        "models": {
            "reasoning": "claude-opus-4-7",
            "classification": "claude-sonnet-4-6",
            "summarization": "claude-haiku-4-5-20251001",
        },
        "pr": {"mode": "github_action", "base_branch": "master", "branch_prefix": "perf/"},
    }
    cfg = config.from_dict(data)
    assert cfg.target.ref == "v1"
    assert cfg.eligibility.allow_pure_conftest_imports is True
    assert cfg.measurement.autoscale_min_instructions == 500_000_000
    assert cfg.budget.max_extraction_turns == 15
    assert cfg.pr.mode == "github_action"


def test_unknown_key_in_section_rejected():
    with pytest.raises(ValueError, match=r"unknown keys in \[measurement\]"):
        config.from_dict({"measurement": {"improvement_threshold": 0.2, "bogus": 1}})


def test_unknown_top_level_section_rejected():
    with pytest.raises(ValueError, match=r"unknown config sections"):
        config.from_dict({"targets": {"repo_url": "x"}})


def test_non_table_section_rejected():
    with pytest.raises(ValueError, match=r"must be a table"):
        config.from_dict({"target": "not-a-table"})


def test_load_from_toml_file(tmp_path: Path):
    toml = textwrap.dedent(
        """
        [target]
        repo_url = "https://example.com/x"
        ref = "release"

        [measurement]
        improvement_threshold = 0.25
        """
    )
    p = tmp_path / "spinwright.toml"
    p.write_text(toml)
    cfg = config.load(p)
    assert cfg.target.repo_url == "https://example.com/x"
    assert cfg.target.ref == "release"
    assert cfg.measurement.improvement_threshold == 0.25
    # Untouched section keeps defaults.
    assert cfg.corpus.dir == "spinwright"


def test_load_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        config.load(tmp_path / "does-not-exist.toml")


def test_config_is_frozen():
    cfg = config.default()
    with pytest.raises(Exception):
        cfg.target.repo_url = "mutated"  # type: ignore[misc]
