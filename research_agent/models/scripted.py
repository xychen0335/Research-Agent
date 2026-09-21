"""Scripted policies for CPU tests and harness verification."""

from __future__ import annotations

import json
from collections.abc import Sequence

from research_agent.contracts import Generation, TOOL_OPEN, TOOL_SEARCH, TOOL_SUBMIT
from research_agent.models.base import encode_text, estimate_tokens


def tool_call(name: str, arguments: dict) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


class ScriptedPolicy:
    def __init__(self, scripts: dict[str, Sequence[str]], *, policy_version: str = "scripted"):
        self.scripts = scripts
        self.policy_version = policy_version
        self._cursors: dict[str, int] = {}

    def _key(self, messages: list[dict[str, str]]) -> str:
        user = next((item["content"] for item in messages if item.get("role") == "user"), "")
        return user.split("\nKnown context:")[0].strip()

    async def generate(self, messages: list[dict[str, str]]) -> Generation:
        key = self._key(messages)
        lines = self.scripts.get(key) or self.scripts.get("*") or [tool_call(TOOL_SUBMIT, {"answer": "", "citations": []})]
        idx = self._cursors.get(key, 0)
        text = lines[min(idx, len(lines) - 1)]
        self._cursors[key] = idx + 1
        token_ids = encode_text(text)
        prompt_blob = "\n".join(item.get("content", "") for item in messages)
        prompt_ids = encode_text(prompt_blob)
        return Generation(
            text=text,
            token_ids=token_ids,
            logprobs=[-0.1] * len(token_ids),
            prompt_token_ids=prompt_ids,
            finish_reason="stop",
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(token_ids),
            usable_for_rl=True,
        )

    def encode_text(self, text: str) -> list[int]:
        return encode_text(text)

    def count_text(self, text: str) -> int:
        return estimate_tokens(text)


def oracle_script(query: str, doc_id: str, end: int, answer: str, citations: list[str]) -> list[str]:
    return [
        tool_call(TOOL_SEARCH, {"query": query}),
        tool_call(TOOL_OPEN, {"doc_id": doc_id, "start": 0, "end": end}),
        tool_call(TOOL_SUBMIT, {"answer": answer, "citations": citations}),
    ]


def no_retrieval_script(answer: str = "unknown") -> list[str]:
    return [tool_call(TOOL_SUBMIT, {"answer": answer, "citations": []})]


def rag_script(query: str, doc_id: str, answer: str, citations: list[str]) -> list[str]:
    return [
        tool_call(TOOL_SEARCH, {"query": query}),
        tool_call(TOOL_OPEN, {"doc_id": doc_id, "start": 0, "end": 0}),
        tool_call(TOOL_SUBMIT, {"answer": answer, "citations": citations}),
    ]
