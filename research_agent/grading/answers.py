"""Answer normalization and dataset scoring protocols."""

from __future__ import annotations

import re
import string

from research_agent.grading.contracts import GradingSpec

PUNCT_TABLE = str.maketrans("", "", string.punctuation)
ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
SPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    value = text.strip().lower()
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    value = ARTICLE_RE.sub(" ", value)
    value = value.translate(PUNCT_TABLE)
    value = SPACE_RE.sub(" ", value).strip()
    return value


def token_set(text: str) -> set[str]:
    return {tok for tok in normalize_answer(text).split(" ") if tok}


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = token_set(prediction)
    gold_tokens = token_set(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def alias_span(prediction: str, spec: GradingSpec) -> str | None:
    """Return the shortest gold alias that appears as a whole-token span in the prediction.

    Used to harvest short SFT targets from padded teacher submits. Evaluation still
    uses exact match and does not award padded answers.
    """
    pred_norm = f" {normalize_answer(prediction)} "
    if not pred_norm.strip():
        return None
    hits: list[str] = []
    for candidate in (spec.answer, *spec.aliases):
        cand_norm = normalize_answer(candidate)
        if cand_norm and f" {cand_norm} " in pred_norm:
            hits.append(candidate)
    if not hits:
        return None
    return min(hits, key=lambda item: (len(normalize_answer(item).split()), len(item)))


def score_answer(prediction: str, spec: GradingSpec) -> tuple[float, str | None]:
    if not spec.answerable:
        unknown = normalize_answer(prediction) in {"unknown", "n/a", "na", "insufficient evidence", "cannot answer"}
        return (1.0 if unknown else 0.0), ("unknown" if unknown else None)
    candidates = (spec.answer, *spec.aliases)
    pred_norm = normalize_answer(prediction)
    for candidate in candidates:
        if pred_norm == normalize_answer(candidate):
            return 1.0, candidate
        if pred_norm and normalize_answer(candidate) and (
            pred_norm in normalize_answer(candidate) or normalize_answer(candidate) in pred_norm
        ):
            # Containment is allowed only when both sides are short entities.
            if max(len(pred_norm.split()), len(normalize_answer(candidate).split())) <= 6:
                if spec.scoring == "exact_match":
                    continue
    if spec.scoring == "f1":
        best = max(f1_score(prediction, candidate) for candidate in candidates)
        matched = max(candidates, key=lambda item: f1_score(prediction, item))
        return best, matched if best > 0 else None
    return 0.0, None
