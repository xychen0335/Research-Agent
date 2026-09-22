"""Tiny corpus for harness tests. Not a training or eval dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_agent.data.prepare import PreparedData, _finalize
from research_agent.grading.contracts import GradingSpec


@dataclass(frozen=True)
class FixtureTask:
    task_id: str
    question: str
    split: str
    spec: GradingSpec
    search_hint: str
    environment_id: str = "test"
    category: str = "genetics"


DOCUMENTS = [
    {
        "doc_id": "bio:rb1",
        "title": "RB1 in retinoblastoma",
        "paragraphs": [
            "The RB1 gene is mutated in childhood retinoblastoma.",
            "RB1 encodes a tumor suppressor protein.",
        ],
        "source": "test-fixture",
        "version": "test",
        "metadata": {},
    },
    {
        "doc_id": "bio:other",
        "title": "Unrelated abstract",
        "paragraphs": ["Insulin is secreted by pancreatic beta cells."],
        "source": "test-fixture",
        "version": "test",
        "metadata": {},
    },
]

TASKS = [
    FixtureTask(
        task_id="bio-001",
        question="Which gene is mutated in childhood retinoblastoma?",
        split="train",
        spec=GradingSpec(
            task_id="bio-001",
            answer="RB1",
            aliases=("Rb1", "RB1 gene"),
            gold_evidence_ids=("bio:rb1:0",),
            support_doc_ids=("bio:rb1",),
            facts=("RB1",),
            category="genetics",
            scoring="exact_match",
            answerable=True,
        ),
        search_hint="childhood retinoblastoma gene",
    ),
    FixtureTask(
        task_id="bio-002",
        question="Which hormone is secreted by pancreatic beta cells?",
        split="train",
        spec=GradingSpec(
            task_id="bio-002",
            answer="insulin",
            aliases=("Insulin",),
            gold_evidence_ids=("bio:other:0",),
            support_doc_ids=("bio:other",),
            facts=("insulin",),
            category="physiology",
            scoring="exact_match",
            answerable=True,
        ),
        search_hint="pancreatic beta cells insulin",
        category="physiology",
    ),
]


def iter_public_tasks() -> list[dict]:
    return [
        {
            "task_id": task.task_id,
            "question": task.question,
            "split": task.split,
            "environment_id": task.environment_id,
            "category": task.category,
        }
        for task in TASKS
    ]


def iter_grading() -> list[dict]:
    rows = []
    for task in TASKS:
        spec = task.spec
        rows.append(
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
            }
        )
    return rows


def write_prepared(output_dir: Path) -> PreparedData:
    return _finalize(
        "test-fixture",
        output_dir,
        iter_public_tasks(),
        iter_grading(),
        list(DOCUMENTS),
        extra_manifest={"note": "Harness unit-test fixture. Not PaperSearchQA."},
    )
