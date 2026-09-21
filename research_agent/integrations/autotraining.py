"""Public AutoTraining request/response adapter. No Tencent SDK, no gold labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_agent.contracts import Budget, Result, TaskInput, to_plain
from research_agent.harness.trajectory import EpisodeRecord


@dataclass
class AutoTrainingRequest:
    request_id: str
    question: str
    research_context: dict[str, Any]
    environment_id: str
    budget: Budget


def parse_request(payload: dict[str, Any]) -> AutoTrainingRequest:
    raw_budget = payload.get("budget") or {}
    budget = Budget(
        max_explore_calls=int(raw_budget.get("max_explore_calls", 6)),
        max_submit_calls=int(raw_budget.get("max_submit_calls", 1)),
        max_observation_chars=int(raw_budget.get("max_observation_chars", 4000)),
        max_context_tokens=int(raw_budget.get("max_context_tokens", 8192)),
        max_generation_tokens=int(raw_budget.get("max_generation_tokens", 1024)),
    )
    context = dict(payload.get("research_context") or {})
    for banned in ("answer", "aliases", "golden_answers", "gold_evidence_ids", "facts"):
        context.pop(banned, None)
    return AutoTrainingRequest(
        request_id=str(payload["request_id"]),
        question=str(payload["question"]),
        research_context=context,
        environment_id=str(payload.get("environment_id") or "default"),
        budget=budget,
    )


def to_task(request: AutoTrainingRequest) -> TaskInput:
    return TaskInput(
        request_id=request.request_id,
        question=request.question,
        environment_id=request.environment_id,
        budget=request.budget,
        research_context=request.research_context,
        task_id=request.request_id,
    )


def to_response(record: EpisodeRecord) -> dict[str, Any]:
    result: Result = record.result
    return {
        "answer": result.answer,
        "claims": [to_plain(claim) for claim in result.claims],
        "citations": result.citations,
        "conditions": result.conditions,
        "unresolved_questions": result.unresolved_questions,
        "status": result.status,
        "trajectory_id": result.trajectory_id,
        "policy_version": result.policy_version,
        "harness_version": result.harness_version,
        "usage": to_plain(result.usage),
    }
