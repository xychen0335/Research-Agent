"""verl custom_reward_function adapter. Scoring lives in grading/; this file is the hook.

verl's RewardManager calls:

    compute_score(data_source, solution_str, ground_truth, extra_info=None)

Configure on the GPU node:

    reward.custom_reward_function.path=research_agent/training/verl/reward_adapter.py
    reward.custom_reward_function.name=compute_score

Gold may appear in extra_info for this function. It must not be copied into policy messages.
"""

from __future__ import annotations

from typing import Any

from research_agent.contracts import TOOL_SUBMIT
from research_agent.grading.answers import score_answer
from research_agent.grading.contracts import GradingSpec, Score
from research_agent.grading.reward import combine_reward, score_episode
from research_agent.harness.loop import parse_tool_calls
from research_agent.harness.trajectory import EpisodeRecord


def spec_from_reward_inputs(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> GradingSpec:
    extra = extra_info if isinstance(extra_info, dict) else {}
    aliases = extra.get("aliases") or extra.get("golden_answers") or ()
    return GradingSpec(
        task_id=str(extra.get("task_id") or extra.get("index") or ""),
        answer=str(ground_truth if ground_truth is not None else extra.get("answer") or ""),
        aliases=tuple(aliases),
        gold_evidence_ids=tuple(extra.get("gold_evidence_ids") or ()),
        support_doc_ids=tuple(extra.get("support_doc_ids") or ()),
        facts=tuple(extra.get("facts") or ()),
        category=str(extra.get("category") or ""),
        scoring=str(extra.get("scoring") or "exact_match"),
        answerable=bool(extra.get("answerable", True)),
        notes=str(extra.get("notes") or ""),
    )


def submitted_answer(solution_str: str, extra_info: dict[str, Any] | None = None) -> str:
    extra = extra_info if isinstance(extra_info, dict) else {}
    explicit = extra.get("submitted_answer")
    if explicit not in (None, ""):
        return str(explicit)
    for action in reversed(parse_tool_calls(solution_str or "")):
        if action.name == TOOL_SUBMIT:
            return str(action.arguments.get("answer") or "")
    return str(solution_str or "")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """verl RewardManager entry. Returns a scalar used as the trajectory reward."""
    extra = extra_info if isinstance(extra_info, dict) else {}
    spec = spec_from_reward_inputs(ground_truth, extra)
    prediction = submitted_answer(str(solution_str or ""), extra)
    answer_score, _matched = score_answer(prediction, spec)
    cost = extra.get("cost")
    if cost is None:
        explore = extra.get("explore_calls")
        max_explore = extra.get("max_explore_calls") or extra.get("max_explore")
        if explore is not None and max_explore:
            cost = min(1.0, float(explore) / float(max_explore))
        else:
            cost = 0.0
    cost_lambda = float(extra.get("cost_lambda") if extra.get("cost_lambda") is not None else kwargs.get("cost_lambda") or 0.0)
    del data_source
    return float(combine_reward(float(answer_score), float(cost), cost_lambda))


def score_record(record: EpisodeRecord, spec: GradingSpec, *, cost_lambda: float = 0.0) -> Score:
    return score_episode(
        record.result,
        spec,
        cost_lambda=cost_lambda,
        max_explore=record.task.budget.max_explore_calls,
    )


def rewards_from_records(
    records: list[EpisodeRecord],
    specs: dict[str, GradingSpec],
    *,
    cost_lambda: float = 0.0,
) -> list[float]:
    rewards: list[float] = []
    for record in records:
        spec = specs[record.task.public_id()]
        rewards.append(score_record(record, spec, cost_lambda=cost_lambda).reward)
    return rewards
