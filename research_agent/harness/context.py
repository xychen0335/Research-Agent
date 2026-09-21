"""Message construction, observation clipping, and context limits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from research_agent.contracts import Observation
from research_agent.harness.state import EpisodeState


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...


def estimate_tokens(text: str) -> int:
    words = [part for part in text.replace("\n", " ").split(" ") if part]
    return max(1, len(words))


class WhitespaceCounter:
    def count_text(self, text: str) -> int:
        return estimate_tokens(text)


def load_system_prompt(path: Path | None = None) -> str:
    prompt_path = path or Path(__file__).resolve().parents[2] / "prompts" / "agent.md"
    return prompt_path.read_text(encoding="utf-8")


def clip_observation(observation: Observation, max_chars: int) -> Observation:
    text = observation.as_text()
    if len(text) <= max_chars:
        return observation
    clipped = dict(observation.content)
    clipped["_clipped"] = True
    raw = str(clipped.get("text") or clipped.get("snippet") or text)
    clipped["text"] = raw[: max(0, max_chars - 80)]
    return Observation(
        tool=observation.tool,
        content=clipped,
        truncated=True,
        error=observation.error,
        error_kind=observation.error_kind,
        infra_failure=observation.infra_failure,
    )


def observation_message(observation: Observation) -> dict[str, str]:
    body = f"<tool_response>\n{observation.as_text()}\n</tool_response>"
    return {"role": "user", "content": body}


def initial_messages(state: EpisodeState, system_prompt: str) -> list[dict[str, str]]:
    context = state.task.research_context
    extra = ""
    if context:
        bits = [f"{k}: {v}" for k, v in context.items() if v not in (None, "")]
        if bits:
            extra = "\nKnown context:\n" + "\n".join(bits)
    user = state.task.question.strip() + extra
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def count_messages(messages: list[dict[str, Any]], counter: TokenCounter) -> int:
    return sum(counter.count_text(str(item.get("content", ""))) for item in messages)


def truncate_messages(
    messages: list[dict[str, Any]],
    counter: TokenCounter,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], bool]:
    if count_messages(messages, counter) <= max_tokens:
        return messages, False
    kept = list(messages)
    # Drop oldest tool responses first. Keep system, original user, and the latest turn.
    idx = 2
    truncated = False
    while idx < len(kept) - 2 and count_messages(kept, counter) > max_tokens:
        if kept[idx].get("role") == "user" and "<tool_response>" in str(kept[idx].get("content", "")):
            kept[idx] = {
                "role": "user",
                "content": "<tool_response>\n{\"truncated\": true, \"note\": \"earlier observation dropped\"}\n</tool_response>",
            }
            truncated = True
        idx += 1
    while len(kept) > 4 and count_messages(kept, counter) > max_tokens:
        # Remove a (assistant, observation) pair after the seed user message.
        del kept[2:4]
        truncated = True
    return kept, truncated
