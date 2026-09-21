"""Generation interface. Token IDs and logprobs are required for RL updates."""

from __future__ import annotations

from typing import Protocol

from research_agent.contracts import Generation
from research_agent.harness.context import estimate_tokens


def encode_text(text: str) -> list[int]:
    """Deterministic stand-in codec for CPU tests. Not a model tokenizer."""
    tokens = [part for part in text.replace("\n", " ").split(" ") if part]
    ids: list[int] = []
    for token in tokens:
        acc = 2166136261
        for char in token:
            acc ^= ord(char)
            acc = (acc * 16777619) & 0xFFFFFFFF
        ids.append(acc % 50_000)
    return ids or [0]


class PolicyModel(Protocol):
    policy_version: str

    async def generate(self, messages: list[dict[str, str]]) -> Generation: ...

    def encode_text(self, text: str) -> list[int]: ...

    def count_text(self, text: str) -> int: ...
