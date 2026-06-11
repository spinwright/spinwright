from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TargetConfig:
    repo_url: str = ""
    # Empty default means "leave the clone on its remote-tracked default branch"
    # — important because repos disagree on whether that's `main` or `master`.
    # Set explicitly to pin a run to a specific ref for reproducibility.
    ref: str = ""


@dataclass(frozen=True)
class CorpusConfig:
    # Repo-rooted directory where extractions, NOTES.md, baselines.json, etc.
    # live. The contents are uniquely spinwright's (setup/run/verify harnesses
    # named after their source nodeid), so a bare `spinwright/` reads as
    # "spinwright's stuff" rather than disguising it under a generic name.
    dir: str = "spinwright"


@dataclass(frozen=True)
class TestSelectionConfig:
    slow_threshold_seconds: float = 0.01
    top_k_candidates: int = 5
    explicit_nodeid: str = ""


@dataclass(frozen=True)
class EligibilityConfig:
    allow_pure_conftest_imports: bool = False


@dataclass(frozen=True)
class MeasurementConfig:
    improvement_threshold: float = 0.20
    walltime_repeats: int = 5
    callgrind_path: str = "valgrind"
    autoscale_min_instructions: int = 1_000_000_000


@dataclass(frozen=True)
class BudgetConfig:
    tokens_per_run: int = 2_000_000
    max_patches_proposed: int = 1  # was 10
    max_wall_clock_minutes: int = 10
    max_extraction_turns: int = 30


@dataclass(frozen=True)
class ModelConfig:
    # The single model used for all LLM agent work (extraction + optimization).
    # Overridable per-invocation via the CLI ``--model`` flag. Anthropic ids
    # work today; this is the seam where OpenAI/Ollama provider routing will be
    # added (e.g. a future ``provider`` / ``base_url`` field alongside it).
    model: str = "claude-opus-4-7"


@dataclass(frozen=True)
class PRConfig:
    mode: Literal["local", "github_action"] = "local"
    base_branch: str = "main"
    branch_prefix: str = "spinwright/"


@dataclass(frozen=True)
class Config:
    target: TargetConfig = field(default_factory=TargetConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    test_selection: TestSelectionConfig = field(default_factory=TestSelectionConfig)
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    pr: PRConfig = field(default_factory=PRConfig)


_SECTIONS: dict[str, type] = {
    "target": TargetConfig,
    "corpus": CorpusConfig,
    "test_selection": TestSelectionConfig,
    "eligibility": EligibilityConfig,
    "measurement": MeasurementConfig,
    "budget": BudgetConfig,
    "models": ModelConfig,
    "pr": PRConfig,
}


def from_dict(data: dict[str, Any]) -> Config:
    unknown_sections = set(data) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(f"unknown config sections: {sorted(unknown_sections)}")
    sections: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        raw = data.get(key, {})
        if not isinstance(raw, dict):
            raise ValueError(
                f"config section [{key}] must be a table, got {type(raw).__name__}"
            )
        unknown = set(raw) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"unknown keys in [{key}]: {sorted(unknown)}")
        sections[key] = cls(**raw)
    return Config(**sections)


def load(path: str | Path) -> Config:
    p = Path(path)
    with p.open("rb") as f:
        data = tomllib.load(f)
    return from_dict(data)


def default() -> Config:
    return Config()
