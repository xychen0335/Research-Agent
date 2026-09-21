"""Event and episode serialization, plus replay from events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research_agent.contracts import (
    STATUS_BUDGET,
    STATUS_COMPLETED,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_TOOL_FAILURE,
    TERMINATION_BUDGET,
    TERMINATION_INFRA,
    TERMINATION_SUBMITTED,
    Claim,
    Event,
    Result,
    TaskInput,
    TokenTrace,
    Usage,
    to_plain,
)
from research_agent.harness.state import EpisodeState


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_event(event_type: str, episode_id: str, payload: dict[str, Any] | None = None) -> Event:
    return Event(
        event_type=event_type,
        timestamp=utc_now(),
        episode_id=episode_id,
        payload=payload or {},
    )


def result_status(state: EpisodeState) -> str:
    if state.termination == TERMINATION_INFRA:
        return STATUS_TOOL_FAILURE
    if state.termination == TERMINATION_SUBMITTED:
        if state.unread_citations and not state.opened_paragraph_ids:
            return STATUS_INSUFFICIENT_EVIDENCE
        if not state.answer:
            return STATUS_INSUFFICIENT_EVIDENCE
        return STATUS_COMPLETED
    if state.termination == TERMINATION_BUDGET:
        return STATUS_BUDGET
    if not state.answer:
        return STATUS_INSUFFICIENT_EVIDENCE
    return STATUS_COMPLETED


def build_result(state: EpisodeState, latency_ms: float) -> Result:
    return Result(
        answer=state.answer,
        claims=list(state.claims),
        citations=list(state.citations),
        conditions=list(state.conditions),
        unresolved_questions=list(state.unresolved_questions),
        status=result_status(state),
        trajectory_id=state.episode_id,
        policy_version=state.policy_version,
        harness_version=state.harness_version,
        environment_version=state.environment_version,
        usage=state.usage(latency_ms),
        termination=state.termination or STATUS_INSUFFICIENT_EVIDENCE,
        unread_citations=list(state.unread_citations),
    )


@dataclass
class EpisodeRecord:
    episode_id: str
    task: TaskInput
    events: list[Event]
    result: Result
    token_trace: TokenTrace
    live: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": to_plain(self.task),
            "events": [to_plain(event) for event in self.events],
            "result": to_plain(self.result),
            "token_trace": to_plain(self.token_trace),
            "live": self.live,
            "messages": self.messages,
        }

    def public_summary(self) -> dict[str, Any]:
        result = to_plain(self.result)
        return {
            "episode_id": self.episode_id,
            "task_id": self.task.public_id(),
            "question": self.task.question,
            "environment_id": self.task.environment_id,
            "live": self.live,
            "result": result,
            "searches": [
                event.payload.get("query")
                for event in self.events
                if event.event_type == "action" and event.payload.get("name") == "search"
            ],
            "opened": [
                event.payload.get("content", {}).get("paragraphs")
                for event in self.events
                if event.event_type == "observation" and event.payload.get("tool") == "open"
            ],
        }


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def replay_result(events: list[dict[str, Any]] | list[Event]) -> dict[str, Any]:
    """Rebuild the public result from recorded events. Does not call a model."""
    end_payload: dict[str, Any] = {}
    for event in events:
        payload = event.payload if isinstance(event, Event) else event.get("payload", {})
        event_type = event.event_type if isinstance(event, Event) else event.get("event_type")
        if event_type == "episode_end":
            end_payload = payload
    return end_payload.get("result", {})


def usage_from_dict(raw: dict[str, Any]) -> Usage:
    return Usage(
        explore_calls=int(raw.get("explore_calls", 0)),
        search_calls=int(raw.get("search_calls", 0)),
        open_calls=int(raw.get("open_calls", 0)),
        submit_calls=int(raw.get("submit_calls", 0)),
        invalid_actions=int(raw.get("invalid_actions", 0)),
        generation_turns=int(raw.get("generation_turns", 0)),
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        observation_chars=int(raw.get("observation_chars", 0)),
        latency_ms=float(raw.get("latency_ms", 0.0)),
        context_truncated=bool(raw.get("context_truncated", False)),
    )


def result_from_dict(raw: dict[str, Any]) -> Result:
    claims = [
        Claim(text=str(item.get("text", "")), citation_ids=list(item.get("citation_ids", [])))
        for item in raw.get("claims", [])
        if isinstance(item, dict)
    ]
    usage_raw = raw.get("usage", {})
    return Result(
        answer=str(raw.get("answer", "")),
        claims=claims,
        citations=list(raw.get("citations", [])),
        conditions=list(raw.get("conditions", [])),
        unresolved_questions=list(raw.get("unresolved_questions", [])),
        status=str(raw.get("status", "")),
        trajectory_id=str(raw.get("trajectory_id", "")),
        policy_version=str(raw.get("policy_version", "")),
        harness_version=str(raw.get("harness_version", "")),
        environment_version=str(raw.get("environment_version", "")),
        usage=usage_from_dict(usage_raw if isinstance(usage_raw, dict) else {}),
        termination=str(raw.get("termination", "")),
        unread_citations=list(raw.get("unread_citations", [])),
    )
