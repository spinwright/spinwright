from __future__ import annotations

import os
from typing import Any, Protocol

from anthropic import Anthropic


class MessagesAPI(Protocol):
    """The narrow slice of the Anthropic SDK's ``messages`` namespace that
    ``dispatch.run_conversation`` actually uses."""

    def create(self, **kwargs: Any) -> Any: ...


class ClientProtocol(Protocol):
    """Structural type for an Anthropic-shaped client. The real ``Anthropic``
    SDK satisfies this; test fakes can satisfy it without subclassing."""

    messages: MessagesAPI


class MissingAPIKeyError(RuntimeError):
    pass


def make_client(api_key: str | None = None) -> Anthropic:
    """Build an Anthropic SDK client.

    API key resolution: explicit arg → ``ANTHROPIC_API_KEY`` env var → error.
    The SDK reads the env var on its own, but we error eagerly here so the
    failure mode is "could not start the run" rather than "first turn failed".
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "no Anthropic API key found — set ANTHROPIC_API_KEY or pass api_key=..."
        )
    return Anthropic(api_key=key)
