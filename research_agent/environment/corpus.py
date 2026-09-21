"""Frozen document snapshot with stable paragraph IDs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: str
    doc_id: str
    index: int
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    paragraphs: tuple[Paragraph, ...]
    source: str = ""
    version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        body = "\n".join(paragraph.text for paragraph in self.paragraphs)
        return f"{self.title}\n{body}"

    def snippet(self, max_chars: int = 280) -> str:
        body = self.paragraphs[0].text if self.paragraphs else ""
        text = f"{self.title}. {body}".strip()
        return text[:max_chars]


def paragraph_id(doc_id: str, index: int) -> str:
    return f"{doc_id}:{index}"


def document_from_dict(raw: dict[str, Any]) -> Document:
    doc_id = str(raw["doc_id"])
    texts = [str(item) for item in raw.get("paragraphs", [])]
    paragraphs = tuple(
        Paragraph(paragraph_id=paragraph_id(doc_id, index), doc_id=doc_id, index=index, text=text)
        for index, text in enumerate(texts)
    )
    return Document(
        doc_id=doc_id,
        title=str(raw.get("title", "")),
        paragraphs=paragraphs,
        source=str(raw.get("source", "")),
        version=str(raw.get("version", "v1")),
        metadata=dict(raw.get("metadata", {})),
    )


class CorpusSnapshot:
    def __init__(self, documents: Iterable[Document], *, snapshot_id: str | None = None):
        self.documents = {doc.doc_id: doc for doc in documents}
        payload = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "version": doc.version,
                "paragraphs": [para.text for para in doc.paragraphs],
            }
            for doc in sorted(self.documents.values(), key=lambda item: item.doc_id)
        ]
        self.version = snapshot_id or sha256_text(canonical_json(payload))[:16]

    def get(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def get_paragraph(self, pid: str) -> Paragraph | None:
        if ":" not in pid:
            return None
        doc_id, _, index_text = pid.rpartition(":")
        doc = self.get(doc_id)
        if doc is None or not index_text.isdigit():
            return None
        index = int(index_text)
        if index < 0 or index >= len(doc.paragraphs):
            return None
        return doc.paragraphs[index]

    def iter_documents(self) -> list[Document]:
        return list(self.documents.values())

    def to_records(self) -> list[dict[str, Any]]:
        records = []
        for doc in self.iter_documents():
            records.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "version": doc.version,
                    "paragraphs": [para.text for para in doc.paragraphs],
                    "metadata": doc.metadata,
                }
            )
        return records

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]], *, snapshot_id: str | None = None) -> "CorpusSnapshot":
        return cls([document_from_dict(item) for item in records], snapshot_id=snapshot_id)

    @classmethod
    def from_jsonl(cls, path: Path) -> "CorpusSnapshot":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls.from_records(records)
