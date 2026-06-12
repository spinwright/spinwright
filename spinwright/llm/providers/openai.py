"""OpenAI Chat Completions provider.

Converts between Spinwright's internal Anthropic-shaped data and OpenAI's
shape on every request/response boundary:

  * Tools             {name, description, input_schema}
                   ↔ {type:"function", function:{name, description, parameters}}
  * System prompt     separate ``system=`` arg
                   ↔ first message ``{role:"system", content:"..."}``
  * Tool-result blocks
       user msg ``{type:"tool_result", tool_use_id, content, is_error}``
                   ↔ separate message ``{role:"tool", tool_call_id, content}``
  * Assistant content
       ``[{type:"text"}, {type:"tool_use", id, name, input}]``
                   ↔ ``message.content`` + ``message.tool_calls[]``
  * stop_reason       end_turn / tool_use / max_tokens / refusal
                   ↔ stop / tool_calls / length / content_filter
  * Usage             input_tokens / output_tokens
                   ↔ prompt_tokens / completion_tokens
  * cache_control     stripped on outgoing payloads (OpenAI auto-caches)

OllamaProvider subclasses this one with a different ``base_url`` — the
openai-compat endpoint at ``$OLLAMA_HOST/v1`` accepts the same payloads.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from spinwright.llm.providers.base import ProviderResponse


_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
    "function_call": "tool_use",  # legacy field; we don't emit but be safe
}


class OpenAIProvider:
    """Provider for OpenAI's Chat Completions API (and openai-compatible
    endpoints such as Ollama, which OllamaProvider routes via base_url)."""

    name = "openai"

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def create_message(
        self,
        *,
        model: str,
        system: str | list[dict],
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        cache_static_prefix: bool,  # noqa: ARG002  (OpenAI auto-caches)
    ) -> ProviderResponse:
        oa_messages = _build_openai_messages(system, messages)
        oa_tools = _build_openai_tools(tools)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oa_messages,
            "max_tokens": max_tokens,
        }
        if oa_tools:
            kwargs["tools"] = oa_tools
        response = self._client.chat.completions.create(**kwargs)
        return _unmarshal_response(response)


# ---------------------------------------------------------------------------
# Outgoing conversions
# ---------------------------------------------------------------------------


def _build_openai_messages(
    system: str | list[dict],
    messages: list[dict],
) -> list[dict]:
    """Build OpenAI's flat ``messages`` list from Anthropic's ``system=`` +
    ``messages``. Anthropic-style tool_result blocks become standalone
    ``role:"tool"`` messages; anything else passes through with cache_control
    stripped."""
    out: list[dict] = []
    out.append({"role": "system", "content": _system_to_text(system)})
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and _is_tool_result_only(content):
            # Each tool_result block becomes its own role:tool message.
            for block in content:
                tc_id = block.get("tool_use_id", "")
                body = block.get("content", "")
                if block.get("is_error"):
                    body = f"[error] {body}"
                out.append({"role": "tool", "tool_call_id": tc_id, "content": body})
            continue
        if role == "assistant":
            text, tool_calls = _split_assistant_blocks(content)
            asst: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                asst["tool_calls"] = tool_calls
            out.append(asst)
            continue
        # Plain user/system/etc.
        out.append({"role": role, "content": _content_to_text(content)})
    return out


def _system_to_text(system: str | list[dict]) -> str:
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _is_tool_result_only(content) -> bool:
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _split_assistant_blocks(content) -> tuple[str, list[dict]]:
    """Split Anthropic assistant content into (text, tool_calls)."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    if not isinstance(content, list):
        return str(content), []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}) or {}),
                    },
                }
            )
    return "".join(text_parts), tool_calls


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _build_openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic ``{name, description, input_schema}`` → OpenAI
    ``{type:"function", function:{name, description, parameters}}``.
    ``cache_control`` markers are silently dropped."""
    out = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object"}),
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Incoming conversions
# ---------------------------------------------------------------------------


def _unmarshal_response(response: Any) -> ProviderResponse:
    choice = response.choices[0]
    message = choice.message
    blocks: list[dict] = []
    if getattr(message, "content", None):
        blocks.append({"type": "text", "text": message.content})
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw_arguments": tc.function.arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": args,
            }
        )
    finish_reason = getattr(choice, "finish_reason", None) or ""
    stop_reason = _STOP_REASON_MAP.get(finish_reason, finish_reason or "unknown")
    # Some servers fail to set finish_reason="tool_calls" even when emitting
    # tool calls — if we see tool blocks, that's the real signal.
    if any(b.get("type") == "tool_use" for b in blocks):
        stop_reason = "tool_use"
    usage = _unmarshal_usage(getattr(response, "usage", None))
    return ProviderResponse(content=blocks, stop_reason=stop_reason, usage=usage)


def _unmarshal_usage(usage: Any) -> dict:
    if usage is None:
        return {}
    # OpenAI uses prompt_tokens/completion_tokens; rename to Anthropic vocab.
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
