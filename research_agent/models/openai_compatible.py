"""OpenAI-compatible teacher and serving backend.

API responses typically lack token IDs, so generations are marked unusable for RL updates.
Ollama's OpenAI layer drops Qwen3.5 thinking into a side field and ignores `think: false`;
the native `/api/chat` path is used when the base URL points at Ollama.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from research_agent.contracts import Generation
from research_agent.models.base import encode_text, estimate_tokens

STOP_STRING = "</tool_call>"
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

OLLAMA_SYSTEM = (
    "You are a research retrieval agent on a frozen document corpus. "
    "Use search, then open a hit, then submit. Do not browse the live web. "
    "Cite only paragraph IDs returned by open. "
    "The submit answer must be a short entity of a few words, not a sentence."
)

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Lexical search over the frozen corpus.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Read original paragraphs from one document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finish with a short entity answer (a few words, not a sentence) and citation paragraph IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "claims": {"type": "array"},
                    "conditions": {"type": "array", "items": {"type": "string"}},
                    "unresolved_questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer", "citations"],
            },
        },
    },
]


def xml_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def native_tool_calls_to_xml(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for call in payload.get("tool_calls") or []:
        fn = call.get("function") or call
        name = str(fn.get("name") or "")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if name:
            chunks.append(xml_tool_call(name, args if isinstance(args, dict) else {}))
    return "\n".join(chunks)


def markdown_tool_to_xml(text: str) -> str:
    if "<tool_call>" in text:
        if not text.rstrip().endswith(STOP_STRING):
            text = text.rstrip() + f"\n{STOP_STRING}"
        return text
    match = JSON_BLOCK_RE.search(text)
    blob = match.group(1) if match else None
    stripped = text.strip()
    if blob is None and stripped.startswith("{") and stripped.endswith("}"):
        blob = stripped
    if not blob:
        return text
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return text
    if not isinstance(obj, dict):
        return text
    name = obj.get("name") or obj.get("tool")
    args = obj.get("arguments")
    if args is None:
        args = {key: value for key, value in obj.items() if key not in {"name", "tool"}}
    if not name:
        return text
    return xml_tool_call(str(name), args if isinstance(args, dict) else {})


def message_text(payload: dict[str, Any]) -> str:
    native = native_tool_calls_to_xml(payload)
    if native:
        return native
    parts = [
        str(payload.get("reasoning") or ""),
        str(payload.get("thinking") or ""),
        str(payload.get("content") or ""),
    ]
    text = "\n".join(part for part in parts if part).strip()
    return markdown_tool_to_xml(text)


def rewrite_ollama_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    from research_agent.harness.loop import parse_tool_calls

    rewritten: list[dict[str, Any]] = []
    replaced = False
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "system" and not replaced:
            rewritten.append({"role": "system", "content": OLLAMA_SYSTEM})
            replaced = True
            continue
        if role == "assistant":
            actions = parse_tool_calls(content)
            if actions:
                rewritten.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": action.name, "arguments": action.arguments},
                            }
                            for action in actions
                        ],
                    }
                )
                continue
        if role == "user" and "<tool_response>" in content:
            inner = content.replace("<tool_response>", "").replace("</tool_response>", "").strip()
            rewritten.append({"role": "tool", "content": inner})
            continue
        rewritten.append(dict(message))
    if not replaced:
        rewritten.insert(0, {"role": "system", "content": OLLAMA_SYSTEM})
    return rewritten


def _looks_like_ollama(base_url: str) -> bool:
    lowered = base_url.lower()
    return "11434" in lowered or "ollama" in lowered


class OpenAICompatiblePolicy:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        policy_version: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 180.0,
        think: bool | None = None,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("RESEARCH_AGENT_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("RESEARCH_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if think is None:
            think = True
        self.think = think
        suffix = "" if self.think else "-nothink"
        self.policy_version = policy_version or f"{model}{suffix}"

    def _ollama_root(self) -> str:
        root = self.base_url
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return root.rstrip("/")

    async def generate(self, messages: list[dict[str, str]]) -> Generation:
        import httpx

        if _looks_like_ollama(self.base_url):
            body: dict[str, Any] = {
                "model": self.model,
                "messages": rewrite_ollama_messages(messages),
                "stream": False,
                "think": self.think,
                "tools": OLLAMA_TOOLS,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": 8192,
                },
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self._ollama_root()}/api/chat", json=body)
                response.raise_for_status()
                data = response.json()
            text = message_text(data.get("message") or {})
            prompt_tokens = int(
                data.get("prompt_eval_count") or estimate_tokens(" ".join(m["content"] for m in messages))
            )
            completion_tokens = int(data.get("eval_count") or estimate_tokens(text))
            finish_reason = str(data.get("done_reason") or "stop")
        else:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
            choice = data["choices"][0]
            text = message_text(choice.get("message") or {})
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(" ".join(m["content"] for m in messages)))
            completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
            finish_reason = str(choice.get("finish_reason") or "stop")
        return Generation(
            text=text.strip(),
            token_ids=None,
            logprobs=None,
            prompt_token_ids=None,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usable_for_rl=False,
        )

    def encode_text(self, text: str) -> list[int]:
        return encode_text(text)

    def count_text(self, text: str) -> int:
        return estimate_tokens(text)
