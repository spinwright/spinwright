from __future__ import annotations

from unittest.mock import patch

import pytest

from spinwright.llm.factory import (
    AmbiguousModelSpecError,
    MissingAPIKeyError,
    make_provider,
    parse_spec,
)


# ---------------------------------------------------------------------------
# parse_spec
# ---------------------------------------------------------------------------


def test_explicit_anthropic_prefix():
    r = parse_spec("anthropic/claude-opus-4-7")
    assert r.provider_name == "anthropic"
    assert r.model_id == "claude-opus-4-7"


def test_explicit_openai_prefix():
    r = parse_spec("openai/gpt-4o")
    assert r.provider_name == "openai"
    assert r.model_id == "gpt-4o"


def test_explicit_ollama_prefix_preserves_tag():
    """Ollama model ids commonly contain a `:<tag>` suffix (e.g. llama3.1:8b).
    The factory splits on the first ``/`` only so the tag survives intact."""
    r = parse_spec("ollama/llama3.1:8b")
    assert r.provider_name == "ollama"
    assert r.model_id == "llama3.1:8b"


def test_heuristic_anthropic_claude():
    r = parse_spec("claude-opus-4-7")
    assert r.provider_name == "anthropic"
    assert r.model_id == "claude-opus-4-7"


def test_heuristic_openai_gpt():
    r = parse_spec("gpt-4o")
    assert r.provider_name == "openai"


def test_heuristic_openai_oN_family():
    assert parse_spec("o1-mini").provider_name == "openai"
    assert parse_spec("o3-mini").provider_name == "openai"
    assert parse_spec("o4-mini").provider_name == "openai"
    assert parse_spec("o1").provider_name == "openai"


def test_bare_unknown_name_is_ambiguous():
    with pytest.raises(AmbiguousModelSpecError, match="ambiguous"):
        parse_spec("llama3.1")


def test_empty_spec_rejected():
    with pytest.raises(AmbiguousModelSpecError, match="empty"):
        parse_spec("")
    with pytest.raises(AmbiguousModelSpecError, match="empty"):
        parse_spec("   ")


def test_provider_prefix_without_model_rejected():
    with pytest.raises(AmbiguousModelSpecError, match="no model id"):
        parse_spec("anthropic/")


def test_unknown_provider_prefix_falls_through_to_heuristic():
    # `foo/whatever` is not a known provider prefix, so we treat the whole
    # thing as a bare name and try heuristics — they don't match, so error.
    with pytest.raises(AmbiguousModelSpecError):
        parse_spec("foo/whatever")


# ---------------------------------------------------------------------------
# make_provider — anthropic happy + key-resolution paths
# ---------------------------------------------------------------------------


def test_make_anthropic_uses_explicit_key():
    with patch("spinwright.llm.providers.anthropic.Anthropic"):
        provider, model_id = make_provider("claude-opus-4-7", api_key="sk-explicit")
    assert provider.name == "anthropic"
    assert model_id == "claude-opus-4-7"


def test_make_anthropic_reads_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("spinwright.llm.providers.anthropic.Anthropic"):
        provider, _ = make_provider("anthropic/claude-opus-4-7")
    assert provider.name == "anthropic"


def test_make_anthropic_missing_key_errors_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        make_provider("claude-opus-4-7")


# ---------------------------------------------------------------------------
# make_provider — openai key resolution
# ---------------------------------------------------------------------------


def test_make_openai_missing_key_errors_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        make_provider("openai/gpt-4o")


# ---------------------------------------------------------------------------
# make_provider — ollama needs no key
# ---------------------------------------------------------------------------


def test_make_ollama_does_not_require_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    # OllamaProvider doesn't exist yet (step 6); just confirm the factory
    # routes there and tries to instantiate it — the ImportError surfaces
    # cleanly when the file lands.
    try:
        provider, model_id = make_provider("ollama/llama3.1:8b")
    except (ModuleNotFoundError, ImportError):
        # Acceptable until step 6 — confirms we DID route to ollama (not
        # erroring on the API-key check first).
        return
    assert provider.name == "ollama"
    assert model_id == "llama3.1:8b"
