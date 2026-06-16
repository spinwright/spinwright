from __future__ import annotations

from unittest.mock import patch

from spinwright.llm.providers.ollama import OllamaProvider, _ollama_base_url


def test_default_base_url_targets_ollama_cloud(monkeypatch):
    """Default routes to Ollama Cloud per docs.ollama.com/cloud — the typical
    deployment. Users on self-hosted servers override via OLLAMA_HOST."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert _ollama_base_url() == "https://ollama.com/v1"


def test_empty_string_ollama_host_treated_as_unset(monkeypatch):
    """Regression: GitHub Action workflows commonly export
    ``OLLAMA_HOST=${{ inputs.ollama_host }}`` with an empty input, so the env
    var is set to ``""``. Without special handling that becomes a scheme-less
    base URL of ``"/v1"`` and httpx raises an inscrutable UnsupportedProtocol
    error. Empty string must fall through to the cloud default."""
    monkeypatch.setenv("OLLAMA_HOST", "")
    assert _ollama_base_url() == "https://ollama.com/v1"


def test_local_self_hosted_via_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    assert _ollama_base_url() == "http://localhost:11434/v1"


def test_ollama_host_env_var_honored(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu.lan:11434")
    assert _ollama_base_url() == "http://gpu.lan:11434/v1"


def test_ollama_host_with_trailing_slash(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu.lan:11434/")
    assert _ollama_base_url() == "http://gpu.lan:11434/v1"


def test_ollama_host_already_includes_v1(monkeypatch):
    """Don't double-append /v1 if the user already wrote it."""
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu.lan:11434/v1")
    assert _ollama_base_url() == "http://gpu.lan:11434/v1"


def test_constructor_passes_base_url_to_openai_sdk(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu.lan:11434")
    with patch("spinwright.llm.providers.openai.OpenAI") as mock_sdk:
        OllamaProvider()
    kwargs = mock_sdk.call_args.kwargs
    assert kwargs["base_url"] == "http://gpu.lan:11434/v1"
    # Sentinel API key is always non-empty (the openai SDK rejects empty).
    assert kwargs["api_key"]


def test_explicit_api_key_overrides_sentinel():
    """Users running Ollama behind an auth proxy can pass an explicit key."""
    with patch("spinwright.llm.providers.openai.OpenAI") as mock_sdk:
        OllamaProvider(api_key="my-proxy-key")
    assert mock_sdk.call_args.kwargs["api_key"] == "my-proxy-key"


def test_ollama_api_key_env_var_honored(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "env-proxy-key")
    with patch("spinwright.llm.providers.openai.OpenAI") as mock_sdk:
        OllamaProvider()
    assert mock_sdk.call_args.kwargs["api_key"] == "env-proxy-key"


def test_provider_name():
    with patch("spinwright.llm.providers.openai.OpenAI"):
        p = OllamaProvider()
    assert p.name == "ollama"
