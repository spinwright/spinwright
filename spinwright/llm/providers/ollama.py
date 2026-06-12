"""Ollama via the openai-compat endpoint.

Default target is **Ollama's hosted cloud** (``https://ollama.com``) — that's
the typical deployment now and matches the convention documented at
https://docs.ollama.com/cloud . Authenticate by setting ``OLLAMA_API_KEY``.

To run against a self-hosted server instead, set ``OLLAMA_HOST`` to its
base URL (e.g. ``http://localhost:11434``); local servers don't check the
auth header so ``OLLAMA_API_KEY`` can stay unset (we send a sentinel string
in that case so the openai SDK doesn't refuse an empty key).

Wire protocol: ``$OLLAMA_HOST/v1/chat/completions`` — the OpenAI Chat
Completions wire format, including tool calling on capable models
(``llama3.1:8b``, ``qwen2.5:7b``, …). We reuse ``OpenAIProvider`` and only
override the base URL — the conversion between Anthropic-style content
blocks and openai-compat shapes is identical for both targets.

Env vars:
    OLLAMA_HOST     Base URL of the server. Default ``https://ollama.com``.
                    Trailing ``/`` and a ``/v1`` suffix are both normalized.
    OLLAMA_API_KEY  Bearer token. Required for hosted Ollama; ignored on
                    local self-hosted (they don't check it).
"""

from __future__ import annotations

import os

from spinwright.llm.providers.openai import OpenAIProvider


_DEFAULT_OLLAMA_HOST = "https://ollama.com"
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
