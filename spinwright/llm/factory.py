"""Parse a model spec and instantiate the right provider with the right key.

A *model spec* is the string the user types: either ``provider/model`` (explicit)
or just a bare model name we can route via heuristic. The factory is the single
seam where multi-provider concerns live — the dispatch loop and CLI never need
to know which provider got picked.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinwright.llm.providers.base import Provider


class MissingAPIKeyError(RuntimeError):
    """Raised when the env var (or explicit arg) needed for a provider is
    missing. We fail eagerly so the user sees the problem before a long-running
    conversation kicks off, not after the first network call."""


class AmbiguousModelSpecError(ValueError):
    """Raised when a bare model name doesn't match any provider heuristic."""


@dataclass(frozen=True)
class ResolvedSpec:
    provider_name: str  # "anthropic" | "openai" | "ollama"
    model_id: str  # The string passed to the upstream API


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


_KNOWN_PROVIDERS = {"anthropic", "openai", "ollama"}


_HEURISTICS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^claude[-_]"), "anthropic"),
    (re.compile(r"^gpt[-_]"), "openai"),
    (re.compile(r"^o[1-9]([-_]|$)"), "openai"),  # o1-mini, o3, o4-mini, …
]


def parse_spec(spec: str) -> ResolvedSpec:
    """Parse ``provider/model`` or a bare name into a ResolvedSpec.

    ``ollama/llama3.1:8b`` works because we split on the *first* ``/`` only —
    the rest of the model id (including any colons or further slashes) passes
    through unchanged.
    """
    if not spec or not spec.strip():
        raise AmbiguousModelSpecError("model spec is empty")
    spec = spec.strip()

    head, sep, tail = spec.partition("/")
    if sep and head in _KNOWN_PROVIDERS:
        if not tail:
            raise AmbiguousModelSpecError(
                f"model spec {spec!r} has provider but no model id"
            )
        return ResolvedSpec(provider_name=head, model_id=tail)

    # Bare name — try heuristics.
    for pattern, provider_name in _HEURISTICS:
        if pattern.match(spec):
            return ResolvedSpec(provider_name=provider_name, model_id=spec)

    raise AmbiguousModelSpecError(
        f"model spec {spec!r} is ambiguous; prefix with `provider/` "
        f"(one of: {', '.join(sorted(_KNOWN_PROVIDERS))})"
    )


# ---------------------------------------------------------------------------
# Provider instantiation
# ---------------------------------------------------------------------------


def make_provider(
    model_spec: str,
    *,
    api_key: str | None = None,
) -> tuple["Provider", str]:
    """Resolve a model spec to ``(provider_instance, model_id)``.

    ``api_key`` overrides any env-var lookup. Pass ``None`` (the usual case)
    to fall back to the provider's standard env var:

      * anthropic → ``ANTHROPIC_API_KEY``
      * openai    → ``OPENAI_API_KEY``
      * ollama    → ``OLLAMA_API_KEY`` (required when targeting Ollama Cloud,
                    which is the default). When ``OLLAMA_HOST`` is set to a
                    self-hosted endpoint, the key is optional — the factory
                    treats explicit-host as "user picked a target that
                    doesn't need auth" and skips the eager check.

    Raises ``MissingAPIKeyError`` (the provider needs a key and none was found)
    or ``AmbiguousModelSpecError`` (the spec can't be routed).
    """
    resolved = parse_spec(model_spec)

    # Order matters: resolve the API key BEFORE importing the provider
    # module so users get "set OPENAI_API_KEY" rather than an ImportError
    # if the SDK isn't installed (or, during staged rollout, before the
    # provider file lands).
    if resolved.provider_name == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "anthropic provider needs an API key — "
                "set ANTHROPIC_API_KEY or pass api_key=..."
            )
        from spinwright.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=key), resolved.model_id

    if resolved.provider_name == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "openai provider needs an API key — "
                "set OPENAI_API_KEY or pass api_key=..."
            )
        from spinwright.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=key), resolved.model_id

    if resolved.provider_name == "ollama":
        # Ollama defaults to the hosted cloud (https://ollama.com), which
        # requires OLLAMA_API_KEY. If the user has overridden the host —
        # presumably pointing at a self-hosted server — we treat that as an
        # explicit "I know what I'm doing" signal and let the key be empty;
        # the provider sends a sentinel value the local server ignores.
        key = api_key or os.environ.get("OLLAMA_API_KEY")
        # ``bool(...)`` treats empty string as False, which is what we want —
        # CI workflows commonly export ``OLLAMA_HOST=${{ inputs.ollama_host }}``
        # with a blank input, and that should NOT count as "user picked a
        # self-hosted endpoint that doesn't need auth."
        host_explicit = bool(os.environ.get("OLLAMA_HOST"))
        if not key and not host_explicit:
            raise MissingAPIKeyError(
                "ollama defaults to Ollama Cloud (https://ollama.com) which "
                "requires an API key — set OLLAMA_API_KEY, or set "
                "OLLAMA_HOST to a self-hosted endpoint that doesn't need auth."
            )
        from spinwright.llm.providers.ollama import OllamaProvider

        return OllamaProvider(api_key=key), resolved.model_id

    # Unreachable — parse_spec only returns known providers, but keep the
    # branch so a future provider can't silently fall through.
    raise AmbiguousModelSpecError(
        f"no provider implementation for {resolved.provider_name!r}"
    )
