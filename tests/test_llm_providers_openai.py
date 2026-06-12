from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from spinwright.llm.providers import openai as openai_provider
from spinwright.llm.providers.openai import (
    OpenAIProvider,
    _build_openai_messages,
    _build_openai_tools,
    _unmarshal_response,
    _unmarshal_usage,
)


# ---------------------------------------------------------------------------
# Outgoing — tools
# ---------------------------------------------------------------------------


def test_tools_converted_to_openai_function_shape():
    anth = [
        {
            "name": "profile_cprofile",
            "description": "Profile run().",
            "input_schema": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        }
    ]
    oa = _build_openai_tools(anth)
    assert oa == [
        {
            "type": "function",
            "function": {
                "name": "profile_cprofile",
                "description": "Profile run().",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                },
            },
        }
    ]


def test_tool_cache_control_marker_stripped():
    """The orchestrator may attach cache_control to tools when targeting
    Anthropic; non-Anthropic providers silently drop the marker."""
    anth = [
        {
            "name": "t",
            "description": "d",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"},
        }
    ]
    oa = _build_openai_tools(anth)
    assert "cache_control" not in oa[0]
    assert "cache_control" not in oa[0]["function"]


# ---------------------------------------------------------------------------
# Outgoing — system + messages
# ---------------------------------------------------------------------------


def test_string_system_becomes_first_system_message():
    msgs = _build_openai_messages("you are sw", [{"role": "user", "content": "hi"}])
    assert msgs[0] == {"role": "system", "content": "you are sw"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_anthropic_system_block_list_with_cache_control_unwraps():
    """Dispatch builds system as a single-text-block list with cache_control
    when targeting Anthropic. OpenAI's path strips the block envelope to bare
    text and discards the marker."""
    system = [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    msgs = _build_openai_messages(system, [])
    assert msgs[0] == {"role": "system", "content": "sys"}


def test_tool_result_user_message_becomes_role_tool():
    anth_messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "p", "input": {"x": 1}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "result body"}
            ],
        },
    ]
    msgs = _build_openai_messages("sys", anth_messages)
    # system + user + assistant + tool (no wrapping user role)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[-1] == {
        "role": "tool",
        "tool_call_id": "tu_1",
        "content": "result body",
    }


def test_tool_result_with_is_error_prefixes_content():
    anth_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_2",
                    "content": "ValueError: nope",
                    "is_error": True,
                }
            ],
        },
    ]
    msgs = _build_openai_messages("sys", anth_messages)
    assert msgs[-1]["content"].startswith("[error]")
    assert "ValueError" in msgs[-1]["content"]


def test_assistant_blocks_split_into_content_plus_tool_calls():
    anth_messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok let me check"},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "profile",
                    "input": {"limit": 5},
                },
            ],
        },
    ]
    msgs = _build_openai_messages("sys", anth_messages)
    asst = msgs[-1]
    assert asst["role"] == "assistant"
    assert asst["content"] == "ok let me check"
    assert len(asst["tool_calls"]) == 1
    tc = asst["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "profile"
    assert json.loads(tc["function"]["arguments"]) == {"limit": 5}


def test_assistant_text_only_yields_no_tool_calls_field():
    msgs = _build_openai_messages(
        "sys",
        [
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
    )
    asst = msgs[-1]
    assert asst["content"] == "hi"
    assert "tool_calls" not in asst


# ---------------------------------------------------------------------------
# Incoming — response unmarshaling
# ---------------------------------------------------------------------------


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunctionCall
    type: str = "function"


@dataclass
class FakeOpenAIMessage:
    content: str | None
    tool_calls: list[FakeToolCall] = field(default_factory=list)


@dataclass
class FakeChoice:
    message: FakeOpenAIMessage
    finish_reason: str


@dataclass
class FakeOpenAIUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class FakeOpenAIResponse:
    choices: list[FakeChoice]
    usage: FakeOpenAIUsage = field(default_factory=FakeOpenAIUsage)


def test_unmarshal_simple_text_response():
    response = FakeOpenAIResponse(
        choices=[
            FakeChoice(
                message=FakeOpenAIMessage(content="hello world", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=FakeOpenAIUsage(prompt_tokens=10, completion_tokens=3),
    )
    pr = _unmarshal_response(response)
    assert pr.content == [{"type": "text", "text": "hello world"}]
    assert pr.stop_reason == "end_turn"
    assert pr.usage == {"input_tokens": 10, "output_tokens": 3}


def test_unmarshal_tool_calls_split_into_blocks():
    response = FakeOpenAIResponse(
        choices=[
            FakeChoice(
                message=FakeOpenAIMessage(
                    content="let me check",
                    tool_calls=[
                        FakeToolCall(
                            id="tu_1",
                            function=FakeFunctionCall(
                                name="profile", arguments='{"limit": 5}'
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )
    pr = _unmarshal_response(response)
    assert pr.stop_reason == "tool_use"
    assert pr.content == [
        {"type": "text", "text": "let me check"},
        {"type": "tool_use", "id": "tu_1", "name": "profile", "input": {"limit": 5}},
    ]


def test_unmarshal_finish_reason_normalized():
    cases = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
    }
    for raw, normalized in cases.items():
        response = FakeOpenAIResponse(
            choices=[
                FakeChoice(
                    message=FakeOpenAIMessage(content="x", tool_calls=[]),
                    finish_reason=raw,
                )
            ],
        )
        assert _unmarshal_response(response).stop_reason == normalized


def test_unmarshal_implicit_tool_use_overrides_finish_reason():
    """Some servers leave finish_reason="stop" even when emitting tool calls;
    if we see tool blocks the actual stop_reason has to be tool_use or the
    dispatch loop will exit prematurely without executing the call."""
    response = FakeOpenAIResponse(
        choices=[
            FakeChoice(
                message=FakeOpenAIMessage(
                    content=None,
                    tool_calls=[
                        FakeToolCall(
                            id="tu_1",
                            function=FakeFunctionCall(name="t", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="stop",  # wrong — server bug
            )
        ],
    )
    pr = _unmarshal_response(response)
    assert pr.stop_reason == "tool_use"


def test_unmarshal_invalid_json_arguments_recovers():
    response = FakeOpenAIResponse(
        choices=[
            FakeChoice(
                message=FakeOpenAIMessage(
                    content=None,
                    tool_calls=[
                        FakeToolCall(
                            id="tu_1",
                            function=FakeFunctionCall(name="t", arguments="{not json"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )
    pr = _unmarshal_response(response)
    assert pr.content[0]["input"]["_raw_arguments"] == "{not json"


def test_unmarshal_usage_handles_missing():
    response = FakeOpenAIResponse(
        choices=[
            FakeChoice(
                message=FakeOpenAIMessage(content="ok", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=FakeOpenAIUsage(),
    )
    assert _unmarshal_response(response).usage == {
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_unmarshal_usage_none():
    assert _unmarshal_usage(None) == {}


# ---------------------------------------------------------------------------
# End-to-end: create_message round-trip with a fake SDK
# ---------------------------------------------------------------------------


class FakeChatCompletions:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.response = response
        self.last_kwargs: dict = {}

    def create(self, **kwargs) -> FakeOpenAIResponse:
        self.last_kwargs = kwargs
        return self.response


class FakeChat:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.completions = FakeChatCompletions(response)


class FakeOpenAISDKClient:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.chat = FakeChat(response)


def test_create_message_round_trip(monkeypatch):
    response = FakeOpenAIResponse(
        choices=[
            FakeChoice(
                message=FakeOpenAIMessage(content="round-trip", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=FakeOpenAIUsage(prompt_tokens=5, completion_tokens=2),
    )
    fake_sdk = FakeOpenAISDKClient(response)
    with patch("spinwright.llm.providers.openai.OpenAI", return_value=fake_sdk):
        p = OpenAIProvider(api_key="sk-test")
    result = p.create_message(
        model="gpt-4o",
        system="sys text",
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "name": "t",
                "description": "d",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        max_tokens=512,
        cache_static_prefix=True,
    )
    kwargs = fake_sdk.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["max_tokens"] == 512
    assert kwargs["messages"][0] == {"role": "system", "content": "sys text"}
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["tools"][0]["type"] == "function"
    assert result.stop_reason == "end_turn"
    assert result.content == [{"type": "text", "text": "round-trip"}]


def test_provider_name():
    with patch("spinwright.llm.providers.openai.OpenAI"):
        p = OpenAIProvider(api_key="sk-test")
    assert p.name == "openai"
