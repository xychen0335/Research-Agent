"""Fixed lexical BM25 retriever. Dense retrieval is out of scope for this version."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from research_agent.environment.corpus import CorpusSnapshot, Document

TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    snippet: str
    score: float
    rank: int


class BM25Index:
    def __init__(self, corpus: CorpusSnapshot, *, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc.doc_id for doc in corpus.iter_documents()]
        self.tokenized: dict[str, list[str]] = {
            doc.doc_id: tokenize(doc.search_text) for doc in corpus.iter_documents()
        }
        self.doc_len = {doc_id: max(1, len(tokens)) for doc_id, tokens in self.tokenized.items()}
        self.avgdl = (sum(self.doc_len.values()) / max(1, len(self.doc_len))) if self.doc_len else 1.0
        df: dict[str, int] = defaultdict(int)
        self.tf: dict[str, Counter[str]] = {}
        for doc_id, tokens in self.tokenized.items():
            counts = Counter(tokens)
            self.tf[doc_id] = counts
            for token in counts:
                df[token] += 1
        n = max(1, len(self.doc_ids))
        self.idf = {token: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0) for token, freq in df.items()}

    def search(self, query: str, topk: int = 3) -> list[SearchHit]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: list[tuple[float, str]] = []
        for doc_id in self.doc_ids:
            tf = self.tf[doc_id]
            dl = self.doc_len[doc_id]
            score = 0.0
            for token in q_tokens:
                if token not in tf:
                    continue
                freq = tf[token]
                idf = self.idf.get(token, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1.0)) / denom
            if score > 0:
                scores.append((score, doc_id))
        scores.sort(key=lambda item: (-item[0], item[1]))
        hits: list[SearchHit] = []
        for rank, (score, doc_id) in enumerate(scores[:topk], start=1):
            doc: Document = self.corpus.documents[doc_id]
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    snippet=doc.snippet(),
                    score=round(score, 6),
                    rank=rank,
                )
            )
        return hits
