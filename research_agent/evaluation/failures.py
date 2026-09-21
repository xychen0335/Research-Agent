"""Failure-mode counts for Base / SFT / GRPO eval runs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from research_agent.grading.contracts import Score
from research_agent.harness.loop import parse_tool_calls
from research_agent.harness.trajectory import EpisodeRecord


def summarize_failures(records: Sequence[EpisodeRecord], scores: Sequence[Score]) -> dict[str, Any]:
    terminations: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    invalid_actions = 0
    search_calls = 0
    open_calls = 0
    submit_rate = 0
    tool_success = 0
    tool_total = 0
    turns = 0
    for record, score in zip(records, scores):
        terminations[record.result.termination] += 1
        invalid_actions += record.result.usage.invalid_actions
        search_calls += record.result.usage.search_calls
        open_calls += record.result.usage.open_calls
        turns += record.result.usage.generation_turns
        if record.result.termination == "submitted":
            submit_rate += 1
        generations = [event for event in record.events if event.event_type == "model_generation"]
        observations = [event for event in record.events if event.event_type == "observation"]
        parsed_any = False
        for event in generations:
            actions = parse_tool_calls(str(event.payload.get("text") or ""))
            if actions:
                parsed_any = True
            else:
                by_reason["no_tool_call"] += 1
        for event in observations:
            tool_total += 1
            if event.payload.get("error"):
                kind = str(event.payload.get("error_kind") or "error")
                by_reason[f"tool_{kind}"] += 1
            else:
                tool_success += 1
        if not parsed_any:
            by_reason["episode_without_parsed_tool"] += 1
        if record.result.termination == "submitted" and score.answer_score < 1.0:
            by_reason["wrong_answer"] += 1
        if record.result.termination == "submitted" and score.legal_citation_rate < 1.0:
            by_reason["illegal_or_empty_citation"] += 1
        if record.result.unread_citations:
            by_reason["unread_citation"] += 1
        if record.result.termination != "submitted":
            by_reason[record.result.termination] += 1
    n = len(records) or 1
    return {
        "n": len(records),
        "termination": dict(terminations),
        "reasons": dict(by_reason),
        "mean_generation_turns": turns / n,
        "mean_invalid_actions": invalid_actions / n,
        "mean_search_calls": search_calls / n,
        "mean_open_calls": open_calls / n,
        "submit_rate": submit_rate / n,
        "tool_success_rate": tool_success / tool_total if tool_total else 0.0,
    }
