"""QASPER conversion. Full-text scores and retrieval-converted scores stay separate.

Reads a local json/jsonl dump. Does not download.
"""

from __future__ import annotations

from typing import Any, Iterable


def _flatten_answers(answers_field: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(answers_field, dict):
        nested = answers_field.get("answer") or answers_field.get("answers") or []
        return _flatten_answers(nested)
    if not isinstance(answers_field, list):
        return texts
    for item in answers_field:
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())
        elif isinstance(item, dict):
            for key in ("free_form_answer", "extractive_spans", "answer"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
                elif isinstance(value, list):
                    texts.extend(str(part).strip() for part in value if str(part).strip())
    return texts


def _paragraphs_from_paper(row: dict[str, Any]) -> list[str]:
    paragraphs: list[str] = []
    abstract = row.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        paragraphs.append(abstract.strip())
    full_text = row.get("full_text") or {}
    paragraphs_field = full_text.get("paragraphs") if isinstance(full_text, dict) else None
    if isinstance(paragraphs_field, list):
        for block in paragraphs_field:
            if isinstance(block, list):
                paragraphs.extend(str(part).strip() for part in block if str(part).strip())
            elif isinstance(block, str) and block.strip():
                paragraphs.append(block.strip())
    return paragraphs or [str(row.get("title") or "")]


def convert_qasper_rows(rows: Iterable[dict[str, Any]], *, split: str) -> tuple[list[dict], list[dict], list[dict]]:
    tasks: list[dict] = []
    grading: list[dict] = []
    documents: dict[str, dict] = {}
    for row in rows:
        paper_id = str(row.get("id") or row.get("paper_id") or "").strip()
        title = str(row.get("title") or "")
        doc_id = f"qasper:{paper_id or title[:40]}"
        paragraphs = _paragraphs_from_paper(row)
        if doc_id not in documents:
            documents[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "paragraphs": paragraphs,
                "source": f"qasper:{paper_id}",
                "version": "v1",
                "metadata": {"paper_id": paper_id, "split": split, "task_kind": "single_paper"},
            }
        questions = row.get("qas") or row.get("questions") or []
        if isinstance(row.get("question"), str):
            questions = [row]
        for q_index, qa in enumerate(questions):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            answers = _flatten_answers(qa.get("answers") or qa.get("answer"))
            primary = answers[0] if answers else ""
            qid = str(qa.get("question_id") or f"{paper_id}-q{q_index}")
            task_id = f"qasper-{split}-{qid}"
            tasks.append(
                {
                    "task_id": task_id,
                    "question": question,
                    "split": split,
                    "environment_id": "qasper",
                    "category": "single_paper",
                }
            )
            grading.append(
                {
                    "task_id": task_id,
                    "answer": primary,
                    "aliases": answers,
                    "gold_evidence_ids": [],
                    "support_doc_ids": [doc_id],
                    "facts": answers[:3],
                    "category": "single_paper",
                    "scoring": "f1",
                    "answerable": bool(primary),
                    "notes": "Report full-text QASPER scores separately from retrieval-converted scores.",
                }
            )
    return tasks, grading, list(documents.values())
