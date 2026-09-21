"""Private grading specs. Never import this from harness, environment, or serving paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GradingSpec:
    task_id: str
    answer: str
    aliases: tuple[str, ...] = ()
    gold_evidence_ids: tuple[str, ...] = ()
    support_doc_ids: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    category: str = ""
    scoring: str = "exact_match"
    answerable: bool = True
    notes: str = ""


@dataclass(frozen=True)
class Score:
    task_id: str
    answer_score: float
    legal_citation_rate: float
    evidence_support: float | None
    reward: float
    matched_alias: str | None
    unread_citations: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)
