"""Dedup, leak, and support-document checks. Labels never go into public task files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_agent.environment.corpus import CorpusSnapshot
from research_agent.environment.retrieval import BM25Index
from research_agent.grading.answers import normalize_answer


FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "aliases",
    "golden_answers",
    "gold_evidence_ids",
    "support_doc_ids",
    "facts",
    "grading",
    "search_hint",
}


@dataclass
class ValidationReport:
    n_tasks: int
    n_documents: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    leak_hits: list[str] = field(default_factory=list)
    unretrievable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "n_documents": self.n_documents,
            "ok": not self.errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "leak_hits": self.leak_hits,
            "unretrievable": self.unretrievable,
        }


RETRIEVAL_AUDIT_MAX_TASKS = 200
RETRIEVAL_AUDIT_MAX_DOCS = 2000


def validate_prepared(
    tasks: list[dict[str, Any]],
    grading: list[dict[str, Any]],
    corpus: CorpusSnapshot,
    *,
    retrieve_k: int = 5,
    retrieval_audit: bool | None = None,
) -> ValidationReport:
    report = ValidationReport(n_tasks=len(tasks), n_documents=len(corpus.documents))
    task_ids = [str(item.get("task_id")) for item in tasks]
    if len(task_ids) != len(set(task_ids)):
        report.errors.append("duplicate task_id in public tasks")
    questions = [str(item.get("question", "")).strip().lower() for item in tasks]
    if len(questions) != len(set(questions)):
        report.warnings.append("duplicate questions")
    grading_by_id = {str(item.get("task_id")): item for item in grading}
    tasks_by_id = {str(item.get("task_id")): item for item in tasks}
    missing = [tid for tid in task_ids if tid not in grading_by_id]
    if missing:
        report.errors.append(f"grading missing for {missing[:8]}")

    for task in tasks:
        leaked = sorted(FORBIDDEN_PUBLIC_KEYS.intersection(task.keys()))
        # search_hint may exist only if null; non-null hints are leaks.
        if task.get("search_hint"):
            leaked.append("search_hint")
        leaked = [key for key in leaked if task.get(key) not in (None, "", [], {})]
        if leaked:
            report.errors.append(f"{task.get('task_id')} public record has {leaked}")

    if retrieval_audit is None:
        retrieval_audit = len(tasks) <= RETRIEVAL_AUDIT_MAX_TASKS and len(corpus.documents) <= RETRIEVAL_AUDIT_MAX_DOCS
    index = BM25Index(corpus) if retrieval_audit else None
    if not retrieval_audit:
        report.warnings.append("BM25 retrieval audit skipped for large corpus")

    for spec in grading:
        task_id = str(spec.get("task_id"))
        for doc_id in spec.get("support_doc_ids") or []:
            if corpus.get(str(doc_id)) is None:
                report.errors.append(f"{task_id} support doc missing: {doc_id}")
        for para_id in spec.get("gold_evidence_ids") or []:
            if corpus.get_paragraph(str(para_id)) is None:
                report.errors.append(f"{task_id} gold evidence missing: {para_id}")
        answer = normalize_answer(str(spec.get("answer") or ""))
        for doc_id in spec.get("support_doc_ids") or []:
            doc = corpus.get(str(doc_id))
            if doc is None:
                continue
            blob = doc.search_text.lower()
            if "golden_answers" in blob or "gold_evidence" in blob:
                report.leak_hits.append(f"{task_id} document {doc_id} contains label keys")
        public = tasks_by_id.get(task_id, {})
        if answer and answer in normalize_answer(str(public.get("question") or "")) and spec.get("answerable", True):
            if len(answer.split()) <= 1 and answer not in {"unknown", "yes", "no"}:
                report.warnings.append(f"{task_id} answer string appears in the question")
        if retrieval_audit and spec.get("answerable", True) and spec.get("support_doc_ids"):
            query = str(public.get("question") or "")
            hits = {hit.doc_id for hit in index.search(query, topk=retrieve_k)}
            support = {str(doc_id) for doc_id in spec.get("support_doc_ids") or []}
            if support and hits.isdisjoint(support):
                report.unretrievable.append(task_id)
                report.warnings.append(f"{task_id} support docs not in BM25 top-{retrieve_k} for the raw question")
    return report
