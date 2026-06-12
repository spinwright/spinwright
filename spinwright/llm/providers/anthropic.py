from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from spinwright.llm.providers.base import ProviderResponse


class AnthropicProvider:
    """Passthrough adapter for the Anthropic SDK.

    The dispatch loop's data shapes are already Anthropic-native, so this
    provider's job is mostly to own the SDK instance and the API key. The
    response is unmarshaled into the normalized ``ProviderResponse`` so the
    rest of dispatch sees the same shape it would for any other provider.
    """

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        # API-key resolution + eager validation happens in ``factory.py``;
        # by the time we get here the key is non-empty.
        self._client = Anthropic(api_key=api_key)

    def create_message(
        self,
        *,
        model: str,
        system: str | list[dict],
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        cache_static_prefix: bool,  # already applied to ``system`` by dispatch
    ) -> ProviderResponse:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        content = [_block_to_dict(b) for b in response.content]
        usage = _usage_to_dict(response.usage)
        return ProviderResponse(
            content=content,
            stop_reason=response.stop_reason or "unknown",
            usage=usage,
        )


def _block_to_dict(block: Any) -> dict:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if isinstance(block, dict):
        return block
    raise TypeError(f"unrecognized Anthropic content block: {type(block).__name__}")


def _usage_to_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return usage
    return {}
