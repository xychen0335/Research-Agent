"""Generate SFT traces in the shared harness. The teacher never reads gold labels."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from collections.abc import Callable

from research_agent.contracts import TOOL_OPEN, TOOL_SUBMIT, TaskInput
from research_agent.grading.answers import alias_span, score_answer
from research_agent.grading.contracts import GradingSpec
from research_agent.grading.reward import score_episode
from research_agent.harness.loop import PolicyModel, ToolDispatcher, parse_tool_calls, run_episode
from research_agent.harness.trajectory import EpisodeRecord


@dataclass
class TeacherFilter:
    require_correct: bool = True
    require_legal_citations: bool = True
    keep_recovery: bool = True
    accept_alias_span: bool = True


@dataclass
class TeacherSample:
    record: EpisodeRecord
    kept: bool
    reason: str
    answer_score: float
    recovery: bool


def _had_failed_then_success(record: EpisodeRecord) -> bool:
    saw_error = False
    for event in record.events:
        if event.event_type == "observation" and event.payload.get("error"):
            saw_error = True
        if event.event_type == "action" and event.payload.get("name") == "submit" and saw_error:
            return True
    return False


async def generate_teacher_traces(
    tasks: list[TaskInput],
    specs: dict[str, GradingSpec],
    model: PolicyModel,
    tools: ToolDispatcher,
    *,
    filters: TeacherFilter | None = None,
    samples_per_task: int = 1,
    stop_when_kept: bool = False,
    on_sample: Callable[[TeacherSample], None] | None = None,
    max_kept: int | None = None,
) -> list[TeacherSample]:
    filters = filters or TeacherFilter()
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be >= 1")
    samples: list[TeacherSample] = []
    for task in tasks:
        if task.split == "test":
            continue
        for _draw in range(samples_per_task):
            record = await run_episode(task, model, tools)
            spec = specs[task.public_id()]
            score = score_episode(record.result, spec, max_explore=task.budget.max_explore_calls)
            recovery = _had_failed_then_success(record)
            kept = True
            reason = "kept"
            short = alias_span(record.result.answer, spec) if filters.accept_alias_span else None
            if filters.require_correct and score.answer_score < 1.0:
                if short and score.legal_citation_rate >= 1.0:
                    record = _shorten_record(record, short)
                    reason = "kept_alias_span"
                else:
                    kept = False
                    reason = "incorrect_answer"
            elif filters.require_legal_citations and score.legal_citation_rate < 1.0:
                kept = False
                reason = "illegal_or_empty_citations"
            elif (not filters.keep_recovery) and recovery:
                kept = False
                reason = "recovery_trace"
            samples.append(
                TeacherSample(
                    record=record,
                    kept=kept,
                    reason=reason,
                    answer_score=score.answer_score,
                    recovery=recovery,
                )
            )
            if on_sample is not None:
                on_sample(samples[-1])
            if stop_when_kept and kept:
                break
        if max_kept is not None and sum(1 for item in samples if item.kept) >= max_kept:
            break
    return samples


def rewrite_submit_answer(messages: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") != "assistant" or TOOL_SUBMIT not in content:
            rewritten.append(message)
            continue
        updated = content
        for action in parse_tool_calls(content):
            if action.name != TOOL_SUBMIT or not action.raw_text:
                continue
            payload = {"name": TOOL_SUBMIT, "arguments": {**action.arguments, "answer": answer}}
            replacement = f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"
            updated = updated.replace(action.raw_text, replacement, 1)
        rewritten.append({**message, "content": updated})
    return rewritten


def _shorten_record(record: EpisodeRecord, answer: str) -> EpisodeRecord:
    return replace(
        record,
        result=replace(record.result, answer=answer),
        messages=rewrite_submit_answer(record.messages, answer),
    )


def _opened_paragraph_ids(messages: list[dict[str, Any]]) -> set[str]:
    opened: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for action in parse_tool_calls(str(message.get("content") or "")):
            if action.name != TOOL_OPEN:
                continue
            doc_id = str(action.arguments.get("doc_id") or "")
            if not doc_id:
                continue
            start = int(action.arguments.get("start") or 0)
            end = int(action.arguments.get("end") or start)
            for index in range(min(start, end), max(start, end) + 1):
                opened.add(f"{doc_id}:{index}")
    return opened


def refilter_teacher_rows(rows: list[dict[str, Any]], specs: dict[str, GradingSpec]) -> list[dict[str, Any]]:
    """Harvest short SFT targets from already-sampled traces. Does not change eval EM."""
    kept: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        spec = specs.get(task_id)
        if spec is None or task_id in seen_tasks:
            continue
        pred = str(row.get("answer") or "")
        em, _ = score_answer(pred, spec)
        short = pred.strip() if em >= 1.0 else alias_span(pred, spec)
        citations = [str(item) for item in (row.get("citations") or [])]
        messages = list(row.get("messages") or [])
        if not short or not citations:
            continue
        opened = _opened_paragraph_ids(messages)
        if any(cite not in opened for cite in citations):
            continue
        if em < 1.0:
            messages = rewrite_submit_answer(messages, short)
        kept.append(
            {
                "task_id": task_id,
                "messages": messages,
                "answer": short,
                "citations": citations,
                "policy_version": row.get("policy_version"),
                "harness_version": row.get("harness_version"),
                "environment_version": row.get("environment_version"),
                "usable_for_rl": row.get("usable_for_rl"),
                "recovery": row.get("recovery"),
                "teacher_filter_reason": "kept" if em >= 1.0 else "kept_alias_span",
            }
        )
        seen_tasks.add(task_id)
    return kept


def tasks_without_kept(tasks: list[TaskInput], kept_rows: list[dict[str, Any]]) -> list[TaskInput]:
    kept_ids = {str(row.get("task_id") or "") for row in kept_rows}
    kept_ids.discard("")
    return [task for task in tasks if task.public_id() not in kept_ids]


def sft_row(sample: TeacherSample) -> dict[str, Any]:
    record = sample.record
    return {
        "task_id": record.task.public_id(),
        "messages": record.messages,
        "answer": record.result.answer,
        "citations": record.result.citations,
        "policy_version": record.result.policy_version,
        "harness_version": record.result.harness_version,
        "environment_version": record.result.environment_version,
        "usable_for_rl": record.token_trace.usable_for_rl,
        "recovery": sample.recovery,
        "teacher_filter_reason": sample.reason,
    }
