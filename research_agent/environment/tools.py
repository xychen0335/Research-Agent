"""search / open / submit schemas and dispatch. Tool payloads contain no gold labels."""

from __future__ import annotations

from typing import Any

from research_agent.contracts import (
    ERROR_EMPTY,
    ERROR_INVALID_PARAMS,
    TOOL_OPEN,
    TOOL_SEARCH,
    TOOL_SUBMIT,
    Action,
    Observation,
)
from research_agent.environment.corpus import CorpusSnapshot
from research_agent.environment.retrieval import BM25Index
from research_agent.harness.state import EpisodeState

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": TOOL_SEARCH,
        "description": "Lexical search over the frozen corpus. Returns top-k document IDs, titles, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": TOOL_OPEN,
        "description": "Open a document and return original paragraphs with stable paragraph IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 0},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": TOOL_SUBMIT,
        "description": "Submit a short answer and citation paragraph IDs, then stop.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "claims": {"type": "array"},
                "conditions": {"type": "array", "items": {"type": "string"}},
                "unresolved_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "citations"],
        },
    },
]


def _query_text(arguments: dict[str, Any]) -> str:
    query = arguments.get("query", "")
    if isinstance(query, list):
        return " ".join(str(item) for item in query if str(item).strip())
    return str(query).strip()


class ToolEnvironment:
    def __init__(self, corpus: CorpusSnapshot, *, search_topk: int = 3, max_open_paragraphs: int = 4):
        self.corpus = corpus
        self.index = BM25Index(corpus)
        self.search_topk = search_topk
        self.max_open_paragraphs = max_open_paragraphs
        self.environment_version = corpus.version
        self.schemas = TOOL_SCHEMAS

    async def execute(self, action: Action, state: EpisodeState) -> Observation:
        if action.name == TOOL_SEARCH:
            return await self._search(action, state)
        if action.name == TOOL_OPEN:
            return await self._open(action, state)
        if action.name == TOOL_SUBMIT:
            return await self._submit(action)
        return Observation(
            tool=action.name,
            content={},
            error=f"unknown tool {action.name}",
            error_kind=ERROR_INVALID_PARAMS,
        )

    async def _search(self, action: Action, state: EpisodeState) -> Observation:
        query = _query_text(action.arguments)
        duplicate = query in state.search_queries and query != ""
        if not query:
            return Observation(
                tool=TOOL_SEARCH,
                content={"query": query, "hits": [], "duplicate": duplicate},
                error="empty query",
                error_kind=ERROR_EMPTY,
            )
        topk = int(action.arguments.get("topk", state.budget.search_topk or self.search_topk))
        topk = max(1, min(topk, 8))
        hits = self.index.search(query, topk=topk)
        payload = [
            {
                "doc_id": hit.doc_id,
                "title": hit.title,
                "snippet": hit.snippet,
                "score": hit.score,
                "rank": hit.rank,
            }
            for hit in hits
        ]
        error = None if payload else "no hits"
        return Observation(
            tool=TOOL_SEARCH,
            content={"query": query, "hits": payload, "duplicate": duplicate},
            error=error,
            error_kind=ERROR_EMPTY if error else None,
        )

    async def _open(self, action: Action, state: EpisodeState) -> Observation:
        doc_id = str(action.arguments.get("doc_id", "")).strip()
        doc = self.corpus.get(doc_id)
        if doc is None:
            return Observation(
                tool=TOOL_OPEN,
                content={"doc_id": doc_id},
                error=f"unknown doc_id {doc_id}",
                error_kind=ERROR_INVALID_PARAMS,
            )
        start = int(action.arguments.get("start", 0) or 0)
        end = int(action.arguments.get("end", start) or start)
        if start < 0 or end < start:
            return Observation(
                tool=TOOL_OPEN,
                content={"doc_id": doc_id, "start": start, "end": end},
                error="invalid paragraph range",
                error_kind=ERROR_INVALID_PARAMS,
            )
        last = min(end, len(doc.paragraphs) - 1, start + self.max_open_paragraphs - 1)
        if start >= len(doc.paragraphs):
            return Observation(
                tool=TOOL_OPEN,
                content={"doc_id": doc_id, "start": start, "end": end, "n_paragraphs": len(doc.paragraphs)},
                error="start out of range",
                error_kind=ERROR_INVALID_PARAMS,
            )
        paragraphs = [
            {"paragraph_id": para.paragraph_id, "index": para.index, "text": para.text}
            for para in doc.paragraphs[start : last + 1]
        ]
        return Observation(
            tool=TOOL_OPEN,
            content={
                "doc_id": doc.doc_id,
                "title": doc.title,
                "start": start,
                "end": last,
                "n_paragraphs": len(doc.paragraphs),
                "paragraphs": paragraphs,
            },
        )

    async def _submit(self, action: Action) -> Observation:
        answer = str(action.arguments.get("answer", "")).strip()
        citations = action.arguments.get("citations", [])
        if not isinstance(citations, list):
            return Observation(
                tool=TOOL_SUBMIT,
                content={},
                error="citations must be a list of paragraph IDs",
                error_kind=ERROR_INVALID_PARAMS,
            )
        unknown = [str(item) for item in citations if self.corpus.get_paragraph(str(item)) is None]
        if unknown:
            return Observation(
                tool=TOOL_SUBMIT,
                content={"unknown_citations": unknown},
                error="citations must be existing paragraph IDs",
                error_kind=ERROR_INVALID_PARAMS,
            )
        return Observation(
            tool=TOOL_SUBMIT,
            content={"accepted": True, "answer": answer, "citations": [str(item) for item in citations]},
        )
