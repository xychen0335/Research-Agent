"""Evidence-first task synthesis. Labels are written only to private grading files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_agent.environment.corpus import CorpusSnapshot, paragraph_id
from research_agent.grading.contracts import GradingSpec


def tasks_from_facts(
    facts: list[dict[str, Any]],
    corpus: CorpusSnapshot,
    *,
    split: str = "synth",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each fact must cite paragraph IDs that exist in the snapshot."""
    tasks: list[dict[str, Any]] = []
    grading: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        evidence = [str(item) for item in fact.get("evidence_paragraph_ids") or []]
        for pid in evidence:
            if corpus.get_paragraph(pid) is None:
                raise ValueError(f"evidence paragraph missing: {pid}")
        task_id = str(fact.get("task_id") or f"{split}-{index:04d}")
        docs = sorted({pid.rsplit(":", 1)[0] for pid in evidence})
        tasks.append(
            {
                "task_id": task_id,
                "question": str(fact["question"]),
                "split": split,
                "environment_id": str(fact.get("environment_id") or "synthetic"),
                "category": str(fact.get("category") or "synth"),
            }
        )
        spec = GradingSpec(
            task_id=task_id,
            answer=str(fact["answer"]),
            aliases=tuple(str(x) for x in fact.get("aliases") or []),
            gold_evidence_ids=tuple(evidence),
            support_doc_ids=tuple(docs),
            facts=(str(fact["answer"]),),
            category=str(fact.get("category") or "synth"),
            scoring=str(fact.get("scoring") or "exact_match"),
            answerable=bool(fact.get("answerable", True)),
            notes=str(fact.get("notes") or ""),
        )
        grading.append(
            {
                "task_id": spec.task_id,
                "answer": spec.answer,
                "aliases": list(spec.aliases),
                "gold_evidence_ids": list(spec.gold_evidence_ids),
                "support_doc_ids": list(spec.support_doc_ids),
                "facts": list(spec.facts),
                "category": spec.category,
                "scoring": spec.scoring,
                "answerable": spec.answerable,
                "notes": spec.notes,
            }
        )
    return tasks, grading


def paragraphs_for_document(doc_id: str, texts: list[str]) -> list[str]:
    return [paragraph_id(doc_id, index) for index, _ in enumerate(texts)]


def dump_private(path: Path, grading: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in grading:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
