"""Frozen planning-model packets and a three-way blind-review export.

The planner is a deterministic extractor over the public Result. Its ability is
fixed so differences come from the research agent input, not from a stronger
planner. Human scores stay empty. This is not measured GPU-hour savings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_agent.contracts import Result, to_plain
from research_agent.harness.trajectory import EpisodeRecord, load_jsonl

PLANNER_VERSION = "frozen-extractor-v1"
PLANNER_BUDGET_TOKENS = 0

PLANNING_PROMPT = """You are an experiment planner. Using only the research-agent result below, list:
1. settings that must be matched to reproduce a claimed gain
2. claims that lack citations
3. missing conditions
Do not assume the agent is correct. If evidence is missing, say so.
"""

HUMAN_RUBRIC = (
    "setting_error",
    "unsupported_claim",
    "missing_condition",
)

CS005_MARKERS = ("extramix", "boostsplit", "fullbench")

DEFAULT_CONDITIONS = (
    ("no_retrieval", "synth-no-retrieval", "无检索 Base"),
    ("fixed_rag", "synth-fixed-rag", "固定检索 RAG"),
    ("trained_agent", "sft-qwen35-4b", "训练后 Research Agent"),
    ("scripted_oracle", "cpu-oracle", "脚本 oracle，不是训练后的 Qwen3.5-4B"),
)


def _result_dict(result: Result | dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return to_plain(result)


def frozen_plan(result: Result | dict[str, Any] | None) -> dict[str, Any]:
    payload = _result_dict(result)
    answer = str(payload.get("answer") or "")
    citations = [str(item) for item in payload.get("citations") or []]
    claims = payload.get("claims") or []
    uncited: list[str] = []
    for claim in claims:
        if isinstance(claim, dict):
            text = str(claim.get("text") or "")
            cites = claim.get("citation_ids") or []
            if text and not cites:
                uncited.append(text)
        elif str(claim).strip():
            uncited.append(str(claim))
    settings = [str(item) for item in payload.get("conditions") or [] if str(item).strip()]
    if answer.strip():
        settings.append(f"agent_answer:{answer.strip()}")
    missing = [str(item) for item in payload.get("unresolved_questions") or [] if str(item).strip()]
    if not citations:
        missing.append("agent result has no citations")
    if not answer.strip() or answer.strip().lower() in {"unknown", "n/a", "na"}:
        missing.append("agent did not submit a usable answer")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "planner_version": PLANNER_VERSION,
        "prompt": PLANNING_PROMPT,
        "settings_to_match": settings,
        "uncited_claims": uncited,
        "missing_conditions": missing,
        "reproduced_answer": answer,
        "cited_ids": citations,
        "research_cost": {
            "explore_calls": usage.get("explore_calls"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_ms": usage.get("latency_ms"),
        },
        "planning_cost": {
            "planner_version": PLANNER_VERSION,
            "planner_tokens": PLANNER_BUDGET_TOKENS,
        },
        "structural_flags": _structural_flags(answer, citations, settings, missing),
    }


def _structural_flags(
    answer: str,
    citations: list[str],
    settings: list[str],
    missing: list[str],
) -> dict[str, bool]:
    blob = " ".join([answer, *citations, *settings, *missing]).lower()
    return {
        "has_citations": bool(citations),
        "mentions_extramix": "extramix" in blob,
        "mentions_boostsplit": "boostsplit" in blob,
        "mentions_fullbench": "fullbench" in blob,
    }


def planning_packet(record: EpisodeRecord, *, planner_model: str, planner_budget_tokens: int) -> dict[str, Any]:
    plan = frozen_plan(record.result)
    return {
        "task_id": record.task.public_id(),
        "question": record.task.question,
        "planner_model": planner_model,
        "planner_budget_tokens": planner_budget_tokens,
        "agent_result": to_plain(record.result),
        "prompt": PLANNING_PROMPT,
        "plan": plan,
        "human_rubric": list(HUMAN_RUBRIC),
        "human_scores": None,
        "note": "Human blind review scores planning errors. This is not measured GPU-hour savings.",
    }


def export_planning_packets(
    records: list[EpisodeRecord],
    *,
    planner_model: str = PLANNER_VERSION,
    planner_budget_tokens: int = PLANNER_BUDGET_TOKENS,
) -> list[dict[str, Any]]:
    return [
        planning_packet(record, planner_model=planner_model, planner_budget_tokens=planner_budget_tokens)
        for record in records
    ]


def _episode_for_task(run_dir: Path, task_id: str) -> dict[str, Any] | None:
    path = Path(run_dir) / "episodes.jsonl"
    if not path.exists():
        return None
    for row in load_jsonl(path):
        task = row.get("task") if isinstance(row.get("task"), dict) else {}
        if str(task.get("task_id") or row.get("task_id") or "") == task_id:
            return row
    return None


def condition_review(
    *,
    name: str,
    run_id: str,
    label: str,
    run_dir: Path | None,
    task_id: str,
) -> dict[str, Any]:
    row = _episode_for_task(run_dir, task_id) if run_dir is not None and run_dir.exists() else None
    if row is None:
        return {
            "name": name,
            "run_id": run_id,
            "label": label,
            "status": "not_run",
            "plan": None,
            "human_scores": None,
            "note": "condition missing; not filled with zeros",
        }
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    question = (row.get("task") or {}).get("question") if isinstance(row.get("task"), dict) else row.get("question")
    return {
        "name": name,
        "run_id": run_id,
        "label": label,
        "status": "ran",
        "question": question,
        "plan": frozen_plan(result),
        "human_rubric": list(HUMAN_RUBRIC),
        "human_scores": None,
        "policy_version": result.get("policy_version"),
    }


def compare_planning_conditions(
    *,
    task_id: str = "cs-005",
    outputs: Path,
    conditions: tuple[tuple[str, str, str], ...] = DEFAULT_CONDITIONS,
) -> dict[str, Any]:
    rows = []
    question = None
    for name, run_id, label in conditions:
        item = condition_review(
            name=name,
            run_id=run_id,
            label=label,
            run_dir=Path(outputs) / run_id,
            task_id=task_id,
        )
        if item.get("question"):
            question = item["question"]
        rows.append(item)
    trained = next((item for item in rows if item["name"] == "trained_agent"), None)
    return {
        "task_id": task_id,
        "question": question,
        "planner_version": PLANNER_VERSION,
        "planner_budget_tokens": PLANNER_BUDGET_TOKENS,
        "human_rubric": list(HUMAN_RUBRIC),
        "conditions": rows,
        "trained_agent_status": (trained or {}).get("status") or "not_run",
        "note": (
            "Frozen extractor planner. Human blind-review scores are empty. "
            "Trained Qwen3.5-4B is not_run until a GPU checkpoint exists. "
            "Not measured GPU-hour savings."
        ),
    }


def write_planning_review(outputs: Path, dest: Path, *, task_id: str = "cs-005") -> dict[str, Any]:
    report = compare_planning_conditions(task_id=task_id, outputs=outputs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
