from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from spinwright.llm.client import ClientProtocol
from spinwright.llm.models import DEFAULT_MAX_TOKENS


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """One tool exposed to the LLM.

    ``input_schema`` is a JSON schema describing the tool's input arguments.
    ``handler`` receives the parsed input dict and returns any JSON-serializable
    value (or raises — exceptions are reported back to the model as is_error).
    """

    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Any]


@dataclass(frozen=True)
class TurnRecord:
    """One turn in the conversation, captured for logging and replay.

    For assistant turns: ``stop_reason`` and ``usage`` are populated.
    For user turns (initial message and tool results): both are None.
    """

    role: str  # "user" | "assistant"
    content: list[dict]
    stop_reason: str | None = None
    usage: dict | None = None


@dataclass
class ConversationResult:
    stop_reason: str
    turns: list[TurnRecord]
    final_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    tool_calls: list[dict] = field(default_factory=list)


class DispatchError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------


def run_conversation(
    client: ClientProtocol,
    *,
    model: str,
    system: str,
    initial_user_message: str | list[dict],
    tools: list[ToolDefinition],
    max_turns: int = 30,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_static_prefix: bool = True,
) -> ConversationResult:
    """Drive a tool-use conversation to completion.

    Loops until the model emits ``stop_reason == "end_turn"``, ``max_turns`` is
    reached, or an unrecoverable stop reason fires. Tool calls in the model's
    response are dispatched through their ``handler`` and the results are fed
    back as a user message.

    ``client`` is any object exposing ``messages.create(**kwargs) -> Message``
    with the Anthropic SDK shape; the real SDK works, and so does any duck-
    typed fake (see tests).

    Caching: Anthropic's cache prefix order is ``tools → system → messages``.
    A single ``cache_control`` marker on the last system block therefore caches
    both tools and system in one breakpoint — that's what ``cache_static_prefix``
    does. We don't add a separate marker on tools because it would create a
    redundant second breakpoint without changing what gets cached.
    """
    handlers = {t.name: t.handler for t in tools}
    system_payload = _build_system(system, cache=cache_static_prefix)
    tools_payload = _build_tools(tools)

    initial_content = (
        [{"type": "text", "text": initial_user_message}]
        if isinstance(initial_user_message, str)
        else initial_user_message
    )
    messages: list[dict] = [{"role": "user", "content": initial_content}]
    turns: list[TurnRecord] = [TurnRecord(role="user", content=initial_content)]
    tool_calls: list[dict] = []

    result = ConversationResult(stop_reason="unknown", turns=turns, final_text="")

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_payload,
            tools=tools_payload,
            messages=messages,
        )
        assistant_content = [_block_to_dict(b) for b in response.content]
        messages.append({"role": "assistant", "content": assistant_content})
        usage_dict = _usage_to_dict(response.usage)
        turns.append(
            TurnRecord(
                role="assistant",
                content=assistant_content,
                stop_reason=response.stop_reason,
                usage=usage_dict,
            )
        )
        _accumulate_usage(result, usage_dict)
        result.final_text = _extract_text(assistant_content)
        result.stop_reason = response.stop_reason or "unknown"

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            # max_tokens, refusal, pause_turn, etc. — surface and exit.
            break

        tool_results = []
        for block in assistant_content:
            if block.get("type") != "tool_use":
                continue
            call_record = {
                "id": block["id"],
                "name": block["name"],
                "input": block.get("input", {}),
            }
            tool_calls.append(call_record)
            tool_results.append(_dispatch_tool(block, handlers))

        if not tool_results:
            # stop_reason said tool_use but we found none — broken response.
            raise DispatchError(
                "stop_reason=tool_use but no tool_use blocks in content"
            )

        user_content = tool_results
        messages.append({"role": "user", "content": user_content})
        turns.append(TurnRecord(role="user", content=user_content))
    else:
        # Loop exited via max_turns rather than break.
        result.stop_reason = "max_turns"

    result.tool_calls = tool_calls
    return result


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------


def _build_system(system: str, *, cache: bool) -> list[dict]:
    block: dict = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _build_tools(tools: list[ToolDefinition]) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def _block_to_dict(block: Any) -> dict:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if isinstance(block, dict):
        return block
    raise DispatchError(f"unrecognized content block type: {type(block).__name__}")


def _usage_to_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return usage
    return {}


def _accumulate_usage(result: ConversationResult, usage: dict) -> None:
    result.input_tokens += usage.get("input_tokens", 0) or 0
    result.output_tokens += usage.get("output_tokens", 0) or 0
    result.cache_creation_input_tokens += (
        usage.get("cache_creation_input_tokens", 0) or 0
    )
    result.cache_read_input_tokens += usage.get("cache_read_input_tokens", 0) or 0


def _extract_text(content: list[dict]) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def _dispatch_tool(block: dict, handlers: dict[str, Callable[[dict], Any]]) -> dict:
    name = block.get("name", "")
    tool_use_id = block.get("id", "")
    tool_input = block.get("input", {}) or {}
    handler = handlers.get(name)
    if handler is None:
        return _tool_error(tool_use_id, f"unknown tool {name!r}")
    try:
        output = handler(tool_input)
    except Exception:
        return _tool_error(tool_use_id, traceback.format_exc())
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": _serialize_tool_output(output),
    }


def _tool_error(tool_use_id: str, message: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": message,
        "is_error": True,
    }


def _serialize_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str, indent=2)
    except (TypeError, ValueError):
        return repr(output)
