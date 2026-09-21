"""Fixed-task evaluation through the shared harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agent.contracts import Budget, TaskInput, to_plain
from research_agent.environment.corpus import CorpusSnapshot
from research_agent.environment.tools import ToolEnvironment
from research_agent.grading.contracts import GradingSpec
from research_agent.grading.reward import score_episode
from research_agent.harness.loop import PolicyModel, run_batch
from research_agent.harness.trajectory import EpisodeRecord, dump_jsonl
from research_agent.evaluation.failures import summarize_failures
from research_agent.evaluation.metrics import summarize_scores


def load_tasks(path: Path, budget: Budget) -> list[TaskInput]:
    tasks: list[TaskInput] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tasks.append(
                TaskInput(
                    request_id=str(row.get("task_id")),
                    task_id=str(row.get("task_id")),
                    question=str(row["question"]),
                    environment_id=str(row.get("environment_id") or "default"),
                    split=str(row.get("split") or "dev"),
                    budget=budget,
                    research_context={
                        k: v
                        for k, v in row.items()
                        if k not in {"task_id", "question", "environment_id", "split", "answer", "aliases"}
                        and v is not None
                    },
                )
            )
    return tasks


def load_grading(path: Path) -> dict[str, GradingSpec]:
    specs: dict[str, GradingSpec] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            spec = GradingSpec(
                task_id=str(row["task_id"]),
                answer=str(row.get("answer") or ""),
                aliases=tuple(row.get("aliases") or []),
                gold_evidence_ids=tuple(row.get("gold_evidence_ids") or []),
                support_doc_ids=tuple(row.get("support_doc_ids") or []),
                facts=tuple(row.get("facts") or []),
                category=str(row.get("category") or ""),
                scoring=str(row.get("scoring") or "exact_match"),
                answerable=bool(row.get("answerable", True)),
                notes=str(row.get("notes") or ""),
            )
            specs[spec.task_id] = spec
    return specs


def strip_research_context(task: TaskInput) -> TaskInput:
    cleaned = {
        key: value
        for key, value in task.research_context.items()
        if key not in {"answer", "aliases", "golden_answers", "gold_evidence_ids", "facts", "search_hint"}
    }
    return TaskInput(
        request_id=task.request_id,
        question=task.question,
        environment_id=task.environment_id,
        budget=task.budget,
        research_context=cleaned,
        task_id=task.task_id,
        split=task.split,
    )


@dataclass
class EvalResult:
    run_id: str
    policy_version: str
    n_tasks: int
    metrics: dict[str, Any]
    records: list[EpisodeRecord]


async def run_evaluation(
    tasks: list[TaskInput],
    specs: dict[str, GradingSpec],
    model: PolicyModel,
    tools: ToolEnvironment,
    *,
    run_id: str,
    output_dir: Path | None = None,
    cost_lambda: float = 0.0,
    max_concurrency: int = 4,
    live: bool = True,
) -> EvalResult:
    public_tasks = [strip_research_context(task) for task in tasks]
    records = await run_batch(public_tasks, model, tools, max_concurrency=max_concurrency, live=live)
    scores = []
    for record in records:
        spec = specs[record.task.public_id()]
        scores.append(
            score_episode(
                record.result,
                spec,
                cost_lambda=cost_lambda,
                max_explore=record.task.budget.max_explore_calls,
            )
        )
    metrics = summarize_scores(scores, records)
    metrics["failures"] = summarize_failures(records, scores)
    metrics["run_id"] = run_id
    metrics["policy_version"] = getattr(model, "policy_version", "unknown")
    metrics["harness_version"] = records[0].result.harness_version if records else None
    metrics["environment_version"] = tools.environment_version
    metrics["live"] = live
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(output_dir / "episodes.jsonl", [record.to_dict() for record in records])
        dump_jsonl(
            output_dir / "events.jsonl",
            [event_row for record in records for event_row in (to_plain(event) for event in record.events)],
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from research_agent.evaluation.planning import export_planning_packets

        dump_jsonl(output_dir / "planning.jsonl", export_planning_packets(records))
    return EvalResult(
        run_id=run_id,
        policy_version=str(metrics["policy_version"]),
        n_tasks=len(records),
        metrics=metrics,
        records=records,
    )


def snapshot_from_dir(prepared_dir: Path) -> tuple[list[TaskInput], dict[str, GradingSpec], ToolEnvironment]:
    public = prepared_dir / "public"
    tasks = load_tasks(public / "tasks.jsonl", Budget())
    specs = load_grading(prepared_dir / "private" / "grading.jsonl")
    corpus = CorpusSnapshot.from_jsonl(public / "corpus.jsonl")
    tools = ToolEnvironment(corpus)
    return tasks, specs, tools
