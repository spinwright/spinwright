"""Ollama via the openai-compat endpoint.

Recent Ollama (≥0.4) exposes ``$OLLAMA_HOST/v1/chat/completions`` that speaks
the OpenAI Chat Completions wire format, including tool calling on models
that support it (``llama3.1:8b``, ``qwen2.5:7b``, etc.). We reuse
``OpenAIProvider`` and only override the base URL — the conversion logic
between Anthropic-style content blocks and openai-compat shapes is the same.

API key: Ollama doesn't require one for local servers, but the openai SDK
demands a non-empty ``api_key`` string. We pass a sentinel; the server
ignores it. ``OLLAMA_API_KEY`` (if set) overrides the sentinel for users
running Ollama behind an auth proxy.
"""

from __future__ import annotations

import os

from spinwright.llm.providers.openai import OpenAIProvider


_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_SENTINEL_KEY = "ollama"  # The openai SDK requires a non-empty string.


def _ollama_base_url() -> str:
    """``$OLLAMA_HOST`` honored, with ``/v1`` appended if missing.

    Real-world OLLAMA_HOST values vary: ``http://host:11434`` (most common),
    ``http://host:11434/`` (trailing slash), or already ``http://host:11434/v1``.
    Normalize so we always end up with exactly one ``/v1`` suffix."""
    raw = os.environ.get("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST).rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return raw + "/v1"


class OllamaProvider(OpenAIProvider):
    """Ollama via the openai-compat endpoint at ``$OLLAMA_HOST/v1``."""

    name = "ollama"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OLLAMA_API_KEY") or _SENTINEL_KEY
        super().__init__(api_key=key, base_url=_ollama_base_url())
