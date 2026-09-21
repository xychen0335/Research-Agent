"""Optional HTTP retrieval service wrapping the same BM25 index."""

from __future__ import annotations

from typing import Any

from research_agent.environment.corpus import CorpusSnapshot
from research_agent.environment.retrieval import BM25Index


def search_payload(index: BM25Index, query: str, topk: int = 3) -> dict[str, Any]:
    hits = index.search(query, topk=topk)
    return {
        "query": query,
        "hits": [
            {
                "doc_id": hit.doc_id,
                "title": hit.title,
                "snippet": hit.snippet,
                "score": hit.score,
                "rank": hit.rank,
            }
            for hit in hits
        ],
    }


def build_app(corpus: CorpusSnapshot):
    from fastapi import FastAPI
    from pydantic import BaseModel

    index = BM25Index(corpus)

    class SearchBody(BaseModel):
        query: str
        topk: int = 3

    app = FastAPI(title="research-agent-retrieval")

    @app.post("/search")
    def search(body: SearchBody) -> dict[str, Any]:
        return search_payload(index, body.query, topk=body.topk)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "environment_version": corpus.version}

    return app
