from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response shape — Anthropic content-block style.

    Every provider converts whatever the underlying SDK returns into this
    shape so the dispatch loop's bookkeeping (text + tool_use block walks,
    cache token counters, etc.) stays identical across providers.

    ``content`` is a list of dicts. Each is either:
      - ``{"type": "text", "text": "..."}``
      - ``{"type": "tool_use", "id": "...", "name": "...", "input": {...}}``

    ``stop_reason`` uses Anthropic's vocabulary: ``end_turn`` | ``tool_use``
    | ``max_tokens`` | ``refusal`` | ``pause_turn``. Providers translate from
    their own (e.g., OpenAI's ``stop`` / ``tool_calls`` / ``length``).

    ``usage`` keys are normalized to Anthropic field names:
    ``input_tokens``, ``output_tokens``, optionally
    ``cache_creation_input_tokens`` / ``cache_read_input_tokens``.
    """

    content: list[dict]
    stop_reason: str
    usage: dict


class Provider(Protocol):
    """The narrow interface that ``dispatch.run_conversation`` depends on.

    A concrete provider owns:
      * SDK construction (auth, base_url, http client tuning)
      * Outgoing request normalization (tools, system, messages)
      * Response unmarshaling into ``ProviderResponse``
      * Anthropic-only knobs (``cache_static_prefix``) are accepted by every
        provider but become no-ops where the upstream API doesn't support
        them.
    """

    name: str  # "anthropic" | "openai" | "ollama"

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        cache_static_prefix: bool,
    ) -> ProviderResponse: ...
