"""RL reward. First version uses answer score only (lambda = 0)."""

from __future__ import annotations

from research_agent.contracts import Result
from research_agent.grading.answers import score_answer
from research_agent.grading.contracts import GradingSpec, Score
from research_agent.grading.evidence import score_evidence


def normalized_cost(result: Result, max_explore: int) -> float:
    if max_explore <= 0:
        return 0.0
    return min(1.0, result.usage.explore_calls / max_explore)


def combine_reward(answer_score: float, cost: float, cost_lambda: float) -> float:
    cost_lambda = min(max(cost_lambda, 0.0), 0.999)
    return answer_score * (1.0 - cost_lambda * cost)


def score_episode(
    result: Result,
    spec: GradingSpec,
    *,
    cost_lambda: float = 0.0,
    max_explore: int | None = None,
) -> Score:
    answer_score, matched = score_answer(result.answer, spec)
    legal, support = score_evidence(result, spec)
    explore_cap = max_explore if max_explore is not None else max(1, result.usage.explore_calls)
    cost = normalized_cost(result, explore_cap)
    reward = combine_reward(answer_score, cost, cost_lambda)
    return Score(
        task_id=spec.task_id,
        answer_score=answer_score,
        legal_citation_rate=legal,
        evidence_support=support,
        reward=reward,
        matched_alias=matched,
        unread_citations=tuple(result.unread_citations),
        extra={"cost": cost, "cost_lambda": cost_lambda, "status": result.status},
    )
