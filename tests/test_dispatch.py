from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import pytest

from spinwright.llm.dispatch import (
    DispatchError,
    ToolDefinition,
    run_conversation,
)
from spinwright.llm.providers.base import ProviderResponse


# ---------------------------------------------------------------------------
# Fake SDK shapes
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
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self) -> None:
        self.responses: list[FakeMessage] = []
        self.calls: list[dict] = []

    def queue(self, *responses: FakeMessage) -> None:
        self.responses.extend(responses)

    def create(self, **kwargs) -> FakeMessage:
        # Snapshot kwargs so subsequent mutations to the messages list don't
        # rewrite history we already recorded.
        self.calls.append(copy.deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("no fake response queued for this call")
        return self.responses.pop(0)


class FakeProvider:
    """Implements the Provider protocol with a queued/canned response stream.

    Tests build FakeMessage instances (the old SDK-shaped objects) and put them
    on ``self.messages.responses``; ``create_message`` pops the next one and
    converts it to a ProviderResponse. ``self.messages.calls`` captures the
    kwargs that came in for assertion."""

    name = "fake"

    def __init__(self) -> None:
        self.messages = FakeMessages()

    def create_message(self, **kwargs) -> ProviderResponse:
        self.messages.calls.append(copy.deepcopy(kwargs))
        if not self.messages.responses:
            raise AssertionError("no fake response queued for this call")
        fake = self.messages.responses.pop(0)
        content = [b.model_dump() for b in fake.content]
        usage = (
            fake.usage.model_dump()
            if hasattr(fake.usage, "model_dump")
            else (fake.usage or {})
        )
        return ProviderResponse(
            content=content,
            stop_reason=fake.stop_reason,
            usage=usage,
        )


# Backwards-compat alias for older tests that imported the class by its old
# name — newer tests should use FakeProvider directly.
FakeClient = FakeProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, handler, *, schema=None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"test tool {name}",
        input_schema=schema
        or {"type": "object", "properties": {}, "additionalProperties": True},
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_turn_end_turn_short_circuits():
    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[FakeText(text="hello there")],
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=10, output_tokens=3),
        )
    )
    result = run_conversation(
        client,
        model="claude-test",
        system="you are a test",
        initial_user_message="hi",
        tools=[],
    )
    assert result.stop_reason == "end_turn"
    assert result.final_text == "hello there"
    assert result.input_tokens == 10
    assert result.output_tokens == 3
    assert len(result.turns) == 2  # initial user + assistant
    assert result.tool_calls == []


def test_tool_use_loop_dispatches_and_continues():
    captured = []

    def echo(tool_input):
        captured.append(tool_input)
        return {"echoed": tool_input}

    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[
                FakeText(text="let me check"),
                FakeToolUse(id="tu_1", name="echo", input={"x": 42}),
            ],
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=20, output_tokens=5),
        ),
        FakeMessage(
            content=[FakeText(text="done!")],
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=30, output_tokens=4),
        ),
    )
    result = run_conversation(
        client,
        model="claude-test",
        system="sys",
        initial_user_message="please echo",
        tools=[_tool("echo", echo)],
    )
    assert result.stop_reason == "end_turn"
    assert result.final_text == "done!"
    assert captured == [{"x": 42}]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "echo"
    # Usage accumulated across both turns.
    assert result.input_tokens == 50
    assert result.output_tokens == 9


def test_tool_handler_exception_returned_as_is_error():
    def boom(_):
        raise ValueError("nope")

    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[FakeToolUse(id="tu_1", name="boom", input={})],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeText(text="ok, recovered")], stop_reason="end_turn"),
    )
    result = run_conversation(
        client,
        model="claude-test",
        system="sys",
        initial_user_message="trigger",
        tools=[_tool("boom", boom)],
    )
    assert result.stop_reason == "end_turn"
    # Look at the second message we sent (the tool_result user turn).
    second_user = client.messages.calls[1]["messages"][-1]
    assert second_user["role"] == "user"
    tr = second_user["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["is_error"] is True
    assert "ValueError" in tr["content"]


def test_unknown_tool_name_returns_is_error():
    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[FakeToolUse(id="tu_1", name="ghost", input={})],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeText(text="oh well")], stop_reason="end_turn"),
    )
    _ = run_conversation(
        client,
        model="claude-test",
        system="sys",
        initial_user_message="trigger",
        tools=[],
    )
    second_user = client.messages.calls[1]["messages"][-1]
    tr = second_user["content"][0]
    assert tr["is_error"] is True
    assert "ghost" in tr["content"]


def test_max_turns_caps_loop():
    client = FakeClient()
    # Queue 50 tool_use responses, but we only allow 3 turns.
    for i in range(50):
        client.messages.queue(
            FakeMessage(
                content=[FakeToolUse(id=f"tu_{i}", name="noop", input={})],
                stop_reason="tool_use",
            )
        )
    result = run_conversation(
        client,
        model="claude-test",
        system="sys",
        initial_user_message="start",
        tools=[_tool("noop", lambda _: "ok")],
        max_turns=3,
    )
    assert result.stop_reason == "max_turns"
    assert len(client.messages.calls) == 3


def test_unrecognized_stop_reason_exits_loop():
    client = FakeClient()
    client.messages.queue(
        FakeMessage(content=[FakeText(text="hit cap")], stop_reason="max_tokens")
    )
    result = run_conversation(
        client,
        model="claude-test",
        system="sys",
        initial_user_message="x",
        tools=[],
    )
    assert result.stop_reason == "max_tokens"
    assert result.final_text == "hit cap"


def test_tool_use_stop_with_no_tool_blocks_raises():
    client = FakeClient()
    client.messages.queue(
        FakeMessage(content=[FakeText(text="confused")], stop_reason="tool_use")
    )
    with pytest.raises(DispatchError):
        run_conversation(
            client,
            model="claude-test",
            system="sys",
            initial_user_message="x",
            tools=[_tool("noop", lambda _: "ok")],
        )


def test_single_cache_breakpoint_on_system_covers_tools_and_system():
    # Anthropic cache prefix order is tools → system → messages, so a single
    # cache_control on the system block caches both tools and system. No
    # marker is placed on tools — it would be a redundant second breakpoint.
    client = FakeClient()
    client.messages.queue(
        FakeMessage(content=[FakeText(text="hi")], stop_reason="end_turn")
    )
    run_conversation(
        client,
        model="claude-test",
        system="cached system prompt",
        initial_user_message="x",
        tools=[
            _tool("first", lambda _: "1"),
            _tool("second", lambda _: "2"),
        ],
    )
    call = client.messages.calls[0]
    assert isinstance(call["system"], list)
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    for t in call["tools"]:
        assert "cache_control" not in t


def test_caching_can_be_disabled():
    client = FakeClient()
    client.messages.queue(
        FakeMessage(content=[FakeText(text="hi")], stop_reason="end_turn")
    )
    run_conversation(
        client,
        model="claude-test",
        system="sys",
        initial_user_message="x",
        tools=[_tool("only", lambda _: "1")],
        cache_static_prefix=False,
    )
    call = client.messages.calls[0]
    assert "cache_control" not in call["system"][0]
    assert "cache_control" not in call["tools"][0]


def test_tool_result_dict_output_is_json_serialized():
    client = FakeClient()
    client.messages.queue(
        FakeMessage(
            content=[FakeToolUse(id="tu_1", name="lookup", input={})],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeText(text="thanks")], stop_reason="end_turn"),
    )
    run_conversation(
        client,
        model="claude-test",
        system="sys",
        initial_user_message="x",
        tools=[_tool("lookup", lambda _: {"answer": 42, "nested": [1, 2]})],
    )
    second_user = client.messages.calls[1]["messages"][-1]
    tr = second_user["content"][0]
    assert tr["type"] == "tool_result"
    assert tr.get("is_error") is not True
    assert '"answer": 42' in tr["content"]
