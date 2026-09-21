"""Citation legality and gold-evidence overlap. Semantic support is not implied by a legal ID."""

from __future__ import annotations

from research_agent.contracts import Result
from research_agent.grading.contracts import GradingSpec


def legal_citation_rate(result: Result) -> float:
    if not result.citations:
        return 0.0
    legal = len(result.citations) - len(result.unread_citations)
    return max(0.0, legal) / len(result.citations)


def evidence_overlap(result: Result, spec: GradingSpec) -> float | None:
    if not spec.gold_evidence_ids:
        return None
    cited = set(result.citations)
    gold = set(spec.gold_evidence_ids)
    if not gold:
        return None
    return len(cited & gold) / len(gold)


def score_evidence(result: Result, spec: GradingSpec) -> tuple[float, float | None]:
    return legal_citation_rate(result), evidence_overlap(result, spec)
