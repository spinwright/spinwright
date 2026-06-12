from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from spinwright.llm.providers.anthropic import AnthropicProvider
from spinwright.llm.providers.base import ProviderResponse


# ---------------------------------------------------------------------------
# Fake SDK shapes that mimic the bits AnthropicProvider touches
# ---------------------------------------------------------------------------


@dataclass
class FakeText:
    text: str
    type: str = "text"

    def model_dump(self, *, exclude_none: bool = True) -> dict:
        return {"type": "text", "text": self.text}


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def model_dump(self, *, exclude_none: bool = True) -> dict:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def model_dump(self, *, exclude_none: bool = True) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str
    usage: FakeUsage


class FakeMessages:
    def __init__(self, response: FakeMessage) -> None:
        self.response = response
        self.last_kwargs: dict = {}

    def create(self, **kwargs) -> FakeMessage:
        self.last_kwargs = kwargs
        return self.response


class FakeAnthropic:
    def __init__(self, response: FakeMessage) -> None:
        self.messages = FakeMessages(response)


def _provider_with(response: FakeMessage) -> tuple[AnthropicProvider, FakeAnthropic]:
    fake = FakeAnthropic(response)
    with patch("spinwright.llm.providers.anthropic.Anthropic", return_value=fake):
        p = AnthropicProvider(api_key="sk-test")
    return p, fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_passes_request_through_unchanged():
    fake_response = FakeMessage(
        content=[FakeText(text="hi")],
        stop_reason="end_turn",
        usage=FakeUsage(input_tokens=10, output_tokens=2),
    )
    p, fake = _provider_with(fake_response)
    result = p.create_message(
        model="claude-opus-4-7",
        system=[
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
        max_tokens=1024,
        cache_static_prefix=True,
    )
    assert isinstance(result, ProviderResponse)
    kwargs = fake.messages.last_kwargs
    # Spinwright passes system/tools/messages through unchanged.
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"][0]["input_schema"] == {"type": "object"}
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_unmarshals_content_and_usage_to_anthropic_shape():
    fake_response = FakeMessage(
        content=[
            FakeText(text="ok"),
            FakeToolUse(id="tu_1", name="profile", input={"limit": 10}),
        ],
        stop_reason="tool_use",
        usage=FakeUsage(input_tokens=50, output_tokens=20, cache_read_input_tokens=100),
    )
    p, _ = _provider_with(fake_response)
    result = p.create_message(
        model="claude-opus-4-7",
        system="sys",
        messages=[],
        tools=[],
        max_tokens=1024,
        cache_static_prefix=False,
    )
    assert result.stop_reason == "tool_use"
    assert result.content == [
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "tu_1", "name": "profile", "input": {"limit": 10}},
    ]
    assert result.usage["input_tokens"] == 50
    assert result.usage["output_tokens"] == 20
    assert result.usage["cache_read_input_tokens"] == 100


def test_unknown_stop_reason_falls_back_to_string():
    fake_response = FakeMessage(
        content=[FakeText(text="?")],
        stop_reason=None,  # SDK occasionally returns None
        usage=FakeUsage(),
    )
    p, _ = _provider_with(fake_response)
    result = p.create_message(
        model="claude-opus-4-7",
        system="sys",
        messages=[],
        tools=[],
        max_tokens=512,
        cache_static_prefix=True,
    )
    assert result.stop_reason == "unknown"


def test_name_attribute_for_routing():
    p, _ = _provider_with(
        FakeMessage(
            content=[FakeText(text="x")], stop_reason="end_turn", usage=FakeUsage()
        )
    )
    assert p.name == "anthropic"
