"""Build public tasks, private grading specs, and corpus snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agent.data.papersearchqa import convert_papersearchqa_rows
from research_agent.data.qasper import convert_qasper_rows
from research_agent.data.validate import validate_prepared
from research_agent.environment.corpus import CorpusSnapshot, canonical_json, sha256_text
from research_agent.paths import PAPERSEARCHQA_RAW, PUBMED_RAW


@dataclass
class PreparedData:
    source: str
    output_dir: Path
    n_tasks: int
    n_documents: int
    corpus_version: str
    report: dict[str, Any]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sha256_text(path.read_text(encoding="utf-8"))


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported dataset file: {path}")
    for key in ("data", "rows", "train", "test"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return [payload]


def prepare_papersearchqa(input_path: Path, output_dir: Path, *, split: str) -> PreparedData:
    rows = _load_json_or_jsonl(input_path)
    tasks, grading, documents = convert_papersearchqa_rows(rows, split=split)
    return _finalize(
        "papersearchqa",
        output_dir,
        tasks,
        grading,
        documents,
        extra_manifest={
            "split": split,
            "input": str(input_path),
            "input_sha256": sha256_text(input_path.read_text(encoding="utf-8")),
        },
    )


def prepare_qasper(input_path: Path, output_dir: Path, *, split: str) -> PreparedData:
    rows = _load_json_or_jsonl(input_path)
    tasks, grading, documents = convert_qasper_rows(rows, split=split)
    return _finalize(
        "qasper",
        output_dir,
        tasks,
        grading,
        documents,
        extra_manifest={
            "split": split,
            "input": str(input_path),
            "input_sha256": sha256_text(input_path.read_text(encoding="utf-8")),
            "note": "Single-paper reading. Retrieval-converted scores must not be mixed with full-text QASPER scores.",
        },
    )


def _finalize(
    source: str,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    grading: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    extra_manifest: dict[str, Any],
    *,
    retrieval_audit: bool | None = None,
) -> PreparedData:
    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = output_dir / "public"
    private_dir = output_dir / "private"
    public_dir.mkdir(exist_ok=True)
    private_dir.mkdir(exist_ok=True)
    corpus = CorpusSnapshot.from_records(documents)
    task_hash = _write_jsonl(public_dir / "tasks.jsonl", tasks)
    doc_hash = _write_jsonl(public_dir / "corpus.jsonl", documents)
    grade_hash = _write_jsonl(private_dir / "grading.jsonl", grading)
    report = validate_prepared(tasks, grading, corpus, retrieval_audit=retrieval_audit)
    manifest = {
        "source": source,
        "n_tasks": len(tasks),
        "n_documents": len(documents),
        "corpus_version": corpus.version,
        "hashes": {
            "tasks.jsonl": task_hash,
            "corpus.jsonl": doc_hash,
            "grading.jsonl": grade_hash,
        },
        "validation": report.as_dict(),
        **extra_manifest,
    }
    (output_dir / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return PreparedData(
        source=source,
        output_dir=output_dir,
        n_tasks=len(tasks),
        n_documents=len(documents),
        corpus_version=corpus.version,
        report=manifest,
    )


def _documents_from_abstracts(abstracts: dict[str, dict], *, split: str) -> list[dict[str, Any]]:
    docs = []
    for pmid, item in abstracts.items():
        doc_id = f"pmid:{pmid}"
        docs.append(
            {
                "doc_id": doc_id,
                "title": item.get("title") or "",
                "paragraphs": [item["abstract"]],
                "source": f"pmid:{pmid}",
                "version": "v1",
                "metadata": {"pmid": pmid, "split": split},
            }
        )
    return docs


def prepare_papersearchqa_dev(
    output_dir: Path,
    *,
    raw_dir: Path | None = None,
    n_train: int = 40,
    n_test: int = 10,
    n_distractors: int = 40,
    seed: int = 20260919,
    fetch_fn=None,
) -> PreparedData:
    from research_agent.data.pubmed import fetch_abstracts
    from research_agent.data.papersearchqa import (
        attach_abstracts,
        convert_papersearchqa_rows,
        load_split_rows,
        parquet_row_count,
        resolve_parquet,
        sample_indices,
    )

    raw_dir = raw_dir or PAPERSEARCHQA_RAW
    train_path = resolve_parquet("train", raw_dir)
    test_path = resolve_parquet("test", raw_dir)
    train_idx = sample_indices(n_train, seed=seed, size=parquet_row_count(train_path))
    test_idx = sample_indices(n_test, seed=seed + 1, size=parquet_row_count(test_path))
    distractor_idx = sample_indices(
        n_distractors, seed=seed + 2, size=parquet_row_count(train_path), exclude=set(train_idx)
    )
    train_rows, train_meta = load_split_rows("train", train_idx, raw_dir)
    test_rows, test_meta = load_split_rows("test", test_idx, raw_dir)
    distractor_rows, _ = load_split_rows("train", distractor_idx, raw_dir)

    pmids = [
        str(row.get("pmid") or "").strip()
        for row in [*train_rows, *test_rows, *distractor_rows]
        if str(row.get("pmid") or "").strip()
    ]
    abstracts = fetch_abstracts(pmids, getter=fetch_fn) if fetch_fn is not None else fetch_abstracts(pmids)
    train_kept, train_excl = attach_abstracts(train_rows, abstracts)
    test_kept, test_excl = attach_abstracts(test_rows, abstracts)
    dist_kept, dist_excl = attach_abstracts(distractor_rows, abstracts)
    train_kept = train_kept[:n_train]
    test_kept = test_kept[:n_test]
    if len(train_kept) < n_train or len(test_kept) < n_test:
        raise RuntimeError(
            f"abstract fetch underfilled the subset: train {len(train_kept)}/{n_train}, test {len(test_kept)}/{n_test}"
        )
    train_pmids = {str(row["pmid"]) for row in train_kept}
    test_pmids = {str(row["pmid"]) for row in test_kept}
    dist_kept = [row for row in dist_kept if str(row["pmid"]) not in train_pmids | test_pmids][:n_distractors]

    train_tasks, train_grading, train_docs = convert_papersearchqa_rows(train_kept, split="train")
    test_tasks, test_grading, test_docs = convert_papersearchqa_rows(test_kept, split="test")
    dist_docs = _documents_from_abstracts(
        {str(row["pmid"]): abstracts[str(row["pmid"])] for row in dist_kept},
        split="distractor",
    )
    docs_by_id = {doc["doc_id"]: doc for doc in [*train_docs, *test_docs, *dist_docs]}
    return _finalize(
        "papersearchqa-dev",
        output_dir,
        train_tasks + test_tasks,
        train_grading + test_grading,
        list(docs_by_id.values()),
        extra_manifest={
            "seed": seed,
            "n_train": len(train_tasks),
            "n_test": len(test_tasks),
            "n_distractors": len(dist_docs),
            "train_parquet": train_meta,
            "test_parquet": test_meta,
            "excluded": train_excl + test_excl + dist_excl,
            "note": (
                "Development subset with gold support abstracts plus distractors. "
                "Not comparable to official 16M-corpus PaperSearchQA numbers. "
                "Test questions must not enter RL."
            ),
        },
    )


def prepare_papersearchqa_full(
    output_dir: Path,
    *,
    raw_dir: Path | None = None,
    with_pubmed_corpus: bool = False,
    pubmed_jsonl: Path | None = None,
    fetch_fn=None,
) -> PreparedData:
    from research_agent.data.pubmed import fetch_abstracts
    from research_agent.data.papersearchqa import (
        attach_abstracts,
        convert_papersearchqa_rows,
        load_split_rows,
        resolve_pubmed_jsonl,
        stream_pubmed_corpus,
    )

    raw_dir = raw_dir or PAPERSEARCHQA_RAW
    train_rows, train_meta = load_split_rows("train", None, raw_dir)
    test_rows, test_meta = load_split_rows("test", None, raw_dir)
    pmids = [
        str(row.get("pmid") or "").strip()
        for row in [*train_rows, *test_rows]
        if str(row.get("pmid") or "").strip()
    ]
    abstracts = fetch_abstracts(pmids, getter=fetch_fn) if fetch_fn is not None else fetch_abstracts(pmids)
    train_kept, train_excl = attach_abstracts(train_rows, abstracts)
    test_kept, test_excl = attach_abstracts(test_rows, abstracts)
    train_tasks, train_grading, train_docs = convert_papersearchqa_rows(train_kept, split="train")
    test_tasks, test_grading, test_docs = convert_papersearchqa_rows(test_kept, split="test")
    docs_by_id = {doc["doc_id"]: doc for doc in [*train_docs, *test_docs]}
    gold_docs = list(docs_by_id.values())
    extra = {
        "n_train": len(train_tasks),
        "n_test": len(test_tasks),
        "train_parquet": train_meta,
        "test_parquet": test_meta,
        "excluded": train_excl + test_excl,
        "note": (
            "Official PaperSearchQA splits. Test questions must not enter RL. "
            "Without --with-pubmed-corpus the search collection is gold abstracts only, not the 16M dump."
        ),
    }
    if not with_pubmed_corpus:
        return _finalize(
            "papersearchqa",
            output_dir,
            train_tasks + test_tasks,
            train_grading + test_grading,
            gold_docs,
            extra_manifest=extra,
        )

    pubmed_path = pubmed_jsonl or resolve_pubmed_jsonl(PUBMED_RAW)
    extra["pubmed_path"] = str(pubmed_path)
    extra["pubmed_bytes"] = pubmed_path.stat().st_size
    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = output_dir / "public"
    private_dir = output_dir / "private"
    public_dir.mkdir(exist_ok=True)
    private_dir.mkdir(exist_ok=True)
    corpus_path = public_dir / "corpus.jsonl"
    stream_report = stream_pubmed_corpus(pubmed_path, corpus_path, gold_documents=gold_docs)
    extra["pubmed_corpus"] = stream_report
    extra["note"] = (
        "Official PaperSearchQA splits plus the 16M PubMed retrieval dump. "
        "Test questions must not enter RL. "
        "In-process BM25 needs enough RAM to index the dump at runtime."
    )
    task_hash = _write_jsonl(public_dir / "tasks.jsonl", train_tasks + test_tasks)
    grade_hash = _write_jsonl(private_dir / "grading.jsonl", train_grading + test_grading)
    corpus = CorpusSnapshot.from_records(gold_docs)
    report = validate_prepared(train_tasks + test_tasks, train_grading + test_grading, corpus, retrieval_audit=False)
    if stream_report["n_appended_gold"]:
        report.warnings.append(f"appended {stream_report['n_appended_gold']} gold abstracts missing from the 16M dump")
    manifest = {
        "source": "papersearchqa",
        "n_tasks": len(train_tasks) + len(test_tasks),
        "n_documents": stream_report["n_documents"],
        "corpus_version": sha256_text(str(stream_report["n_documents"]))[:16],
        "hashes": {
            "tasks.jsonl": task_hash,
            "corpus.jsonl": "streamed-pubmed",
            "grading.jsonl": grade_hash,
        },
        "validation": report.as_dict(),
        **extra,
    }
    (output_dir / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return PreparedData(
        source="papersearchqa",
        output_dir=output_dir,
        n_tasks=len(train_tasks) + len(test_tasks),
        n_documents=int(stream_report["n_documents"]),
        corpus_version=str(manifest["corpus_version"]),
        report=manifest,
    )


def prepare_source(
    source: str,
    output_dir: Path,
    *,
    input_path: Path | None = None,
    split: str = "dev",
    n_train: int = 40,
    n_test: int = 10,
    n_distractors: int = 40,
    seed: int = 20260919,
    with_pubmed_corpus: bool = False,
) -> PreparedData:
    if source in {"papersearchqa-dev", "psqa-dev"}:
        return prepare_papersearchqa_dev(
            output_dir,
            n_train=n_train,
            n_test=n_test,
            n_distractors=n_distractors,
            seed=seed,
        )
    if source in {"papersearchqa", "psqa"}:
        if input_path is not None:
            return prepare_papersearchqa(input_path, output_dir, split=split)
        return prepare_papersearchqa_full(output_dir, with_pubmed_corpus=with_pubmed_corpus)
    if source == "qasper":
        if input_path is None:
            raise ValueError("qasper requires --input")
        return prepare_qasper(input_path, output_dir, split=split)
    raise ValueError(f"unknown source {source}")
