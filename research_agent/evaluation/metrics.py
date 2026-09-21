"""Metric aggregation, cost curves, and paired bootstrap intervals."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

from research_agent.grading.contracts import Score
from research_agent.harness.trajectory import EpisodeRecord


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    if len(a) != len(b) or not a:
        return {"mean_diff": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    samples: list[float] = []
    n = len(diffs)
    for _ in range(n_boot):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(draw))
    samples.sort()
    low_i = int(math.floor(alpha / 2 * n_boot))
    high_i = int(math.floor((1 - alpha / 2) * n_boot))
    high_i = min(n_boot - 1, high_i)
    return {
        "mean_diff": mean(diffs),
        "low": samples[low_i],
        "high": samples[high_i],
        "n": float(n),
    }


def summarize_scores(scores: Sequence[Score], records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    answer = [item.answer_score for item in scores]
    legal = [item.legal_citation_rate for item in scores]
    support = [item.evidence_support for item in scores if item.evidence_support is not None]
    rewards = [item.reward for item in scores]
    explore = [record.result.usage.explore_calls for record in records]
    tokens = [record.result.usage.completion_tokens for record in records]
    latency = [record.result.usage.latency_ms for record in records]
    submitted = sum(1 for record in records if record.result.termination == "submitted")
    failed = sum(1 for record in records if record.result.status != "completed")
    by_budget: dict[str, list[float]] = {"2": [], "4": [], "6": []}
    for score, record in zip(scores, records):
        used = record.result.usage.explore_calls
        if used <= 2:
            by_budget["2"].append(score.answer_score)
        if used <= 4:
            by_budget["4"].append(score.answer_score)
        by_budget["6"].append(score.answer_score)
    return {
        "n": len(scores),
        "answer_em": mean(answer),
        "legal_citation_rate": mean(legal),
        "evidence_support": mean(support) if support else None,
        "reward": mean(rewards),
        "mean_explore_calls": mean(explore),
        "mean_completion_tokens": mean(tokens),
        "mean_latency_ms": mean(latency),
        "submit_rate": submitted / len(records) if records else 0.0,
        "non_completed_rate": failed / len(records) if records else 0.0,
        "cost_curve_em": {k: mean(v) for k, v in by_budget.items()},
        "unrun": False,
    }


def compare_runs(scores_a: Sequence[Score], scores_b: Sequence[Score]) -> dict[str, Any]:
    a = [item.answer_score for item in scores_a]
    b = [item.answer_score for item in scores_b]
    return {"answer_em": paired_bootstrap_ci(a, b)}
