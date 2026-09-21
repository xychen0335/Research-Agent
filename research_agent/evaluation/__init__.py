"""Evaluation package. Runner and baselines stay lazy to keep CPU imports cheap."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaselineName",
    "EvalResult",
    "compare_frozen_runs",
    "run_baseline",
    "run_evaluation",
    "summarize_failures",
    "summarize_scores",
    "write_compare_report",
]


def __getattr__(name: str) -> Any:
    if name in {"compare_frozen_runs", "write_compare_report"}:
        from research_agent.evaluation.compare import compare_frozen_runs, write_compare_report

        return compare_frozen_runs if name == "compare_frozen_runs" else write_compare_report
    if name in {"BaselineName", "run_baseline"}:
        from research_agent.evaluation.baselines import BaselineName, run_baseline

        return BaselineName if name == "BaselineName" else run_baseline
    if name in {"EvalResult", "run_evaluation"}:
        from research_agent.evaluation.runner import EvalResult, run_evaluation

        return EvalResult if name == "EvalResult" else run_evaluation
    if name == "summarize_failures":
        from research_agent.evaluation.failures import summarize_failures

        return summarize_failures
    if name == "summarize_scores":
        from research_agent.evaluation.metrics import summarize_scores

        return summarize_scores
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
