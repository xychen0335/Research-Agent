"""PaperSearchQA row conversion and official-split sampling.

Official location: https://huggingface.co/datasets/jmhb/PaperSearchQA
Hub revision pinned below. License named on the Hub README: MIT for the QA set.
Corpus: jmhb/pubmed_bioasq_2022, derived from allMeSH / BioASQ 2022.
The 16 million abstract dump is not downloaded here. Development subsets keep
gold support abstracts plus fixed distractors.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Iterable

SOURCE_URL = "https://huggingface.co/datasets/jmhb/PaperSearchQA"
CORPUS_URL = "https://huggingface.co/datasets/jmhb/pubmed_bioasq_2022"
CORPUS_REPO = "jmhb/pubmed_bioasq_2022"
CORPUS_JSONL = "data/corpus/pubmed.jsonl"
LICENSE = "MIT (QA set, per Hub README); PubMed corpus derived from allMeSH/BioASQ 2022"
HF_REVISION = "563d32ebcf5a8081ed67abe4f7afe0ae614be1e1"
SPLIT_SIZES = {"train": 54907, "test": 5000}
PARQUET_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def convert_papersearchqa_rows(rows: Iterable[dict[str, Any]], *, split: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Convert official rows into public tasks, grading specs, and abstract documents.

    Gold answers stay in grading records only. Documents contain title and abstract text.
    """
    tasks: list[dict] = []
    grading: list[dict] = []
    documents: dict[str, dict] = {}
    for index, row in enumerate(rows):
        pmid = str(row.get("pmid") or "").strip()
        suffix = pmid if pmid else f"{index:05d}"
        task_id = f"psqa-{split}-{suffix}-{index:05d}"
        doc_id = f"pmid:{pmid}" if pmid else f"psqa-doc-{index:05d}"
        title = str(row.get("paper_title") or row.get("fetched_title") or "")
        abstract = str(row.get("abstract") or row.get("context") or "").strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        aliases = [str(item) for item in row.get("golden_answers") or [] if str(item).strip()]
        if answer and answer not in aliases:
            aliases = [answer, *aliases]
        tasks.append(
            {
                "task_id": task_id,
                "question": question,
                "split": split,
                "environment_id": "papersearchqa",
                "category": str(row.get("cat") or ""),
            }
        )
        grading.append(
            {
                "task_id": task_id,
                "answer": answer,
                "aliases": aliases,
                "gold_evidence_ids": [f"{doc_id}:0"] if abstract else [],
                "support_doc_ids": [doc_id] if pmid else [],
                "facts": [answer],
                "category": str(row.get("cat") or ""),
                "scoring": "exact_match",
                "answerable": True,
                "notes": f"pmid={pmid}",
            }
        )
        if doc_id not in documents and (abstract or title):
            documents[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "paragraphs": [abstract or title],
                "source": f"pmid:{pmid}",
                "version": f"papersearchqa-{HF_REVISION[:8]}",
                "metadata": {"pmid": pmid, "split": split, "hf_revision": HF_REVISION},
            }
    return tasks, grading, list(documents.values())


def download_parquet(split: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/datasets/jmhb/PaperSearchQA/resolve/{HF_REVISION}/"
        f"{PARQUET_FILES[split]}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "research-agent/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    dest.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _rows_from_table(table, indices: list[int]) -> list[dict[str, Any]]:
    subset = table.take(indices)
    columns = subset.to_pydict()
    n = len(next(iter(columns.values()))) if columns else 0
    rows = []
    for i in range(n):
        row = {key: values[i] for key, values in columns.items()}
        answers = row.get("golden_answers")
        if answers is None:
            row["golden_answers"] = []
        else:
            row["golden_answers"] = [str(item) for item in list(answers)]
        rows.append(row)
    return rows


def load_parquet_rows(path: Path, indices: list[int] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if indices is None:
        indices = list(range(table.num_rows))
    return _rows_from_table(table, indices)


def fetch_hf_rows(split: str, indices: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in indices:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset=jmhb/PaperSearchQA&config=default&split={split}&offset={index}&length=1"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "research-agent/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        item = (payload.get("rows") or [{}])[0].get("row") or {}
        rows.append(item)
    return rows


def sample_indices(split: str, n: int, *, seed: int, exclude: set[int] | None = None) -> list[int]:
    size = SPLIT_SIZES[split]
    pool = [i for i in range(size) if not exclude or i not in exclude]
    rng = random.Random(seed)
    if n > len(pool):
        raise ValueError(f"cannot sample {n} from {split} pool {len(pool)}")
    return sorted(rng.sample(pool, n))


def load_split_rows(
    split: str,
    indices: list[int] | None,
    raw_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parquet_path = raw_dir / PARQUET_FILES[split].replace("/", "_")
    meta: dict[str, Any] = {"path": str(parquet_path), "sha256": "", "loader": "", "n_rows": 0}
    try:
        if not parquet_path.exists():
            meta["sha256"] = download_parquet(split, parquet_path)
        else:
            meta["sha256"] = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        rows = load_parquet_rows(parquet_path, indices)
        meta["loader"] = "parquet"
        meta["n_rows"] = len(rows)
        expected = SPLIT_SIZES[split]
        if indices is None and len(rows) != expected:
            meta["row_count_note"] = f"expected {expected}, got {len(rows)}"
        return rows, meta
    except Exception:
        if indices is None:
            raise
        rows = fetch_hf_rows(split, indices)
        meta["loader"] = "datasets-server"
        meta["n_rows"] = len(rows)
        return rows, meta


def attach_abstracts(
    rows: list[dict[str, Any]],
    abstracts: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        pmid = str(row.get("pmid") or "").strip()
        abs_row = abstracts.get(pmid)
        if not abs_row:
            excluded.append({"pmid": pmid, "reason": "abstract_missing"})
            continue
        enriched = dict(row)
        enriched["abstract"] = abs_row["abstract"]
        if not enriched.get("paper_title"):
            enriched["paper_title"] = abs_row.get("title") or ""
        enriched["fetched_title"] = abs_row.get("title") or ""
        kept.append(enriched)
    return kept, excluded


def convert_pubmed_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Map a PubMed dump row into the local corpus schema."""
    pmid = str(raw.get("pmid") or raw.get("id") or raw.get("docid") or "").strip()
    pmid = pmid.removeprefix("pmid:").removeprefix("PMID:")
    title = str(raw.get("title") or "").strip()
    abstract = str(raw.get("abstract") or raw.get("text") or "").strip()
    contents = str(raw.get("contents") or raw.get("content") or "").strip()
    if not abstract and contents:
        if title and contents.startswith(title):
            abstract = contents[len(title) :].lstrip(" .")
        elif not title and ". " in contents[:240]:
            title, _, abstract = contents.partition(". ")
        else:
            abstract = contents
    if not pmid or not (abstract or title):
        return None
    return {
        "doc_id": f"pmid:{pmid}",
        "title": title,
        "paragraphs": [abstract or title],
        "source": f"pmid:{pmid}",
        "version": "pubmed-bioasq-2022",
        "metadata": {"pmid": pmid, "corpus": CORPUS_URL},
    }


def download_pubmed_jsonl(dest: Path) -> str:
    """Stream the official 16M abstract dump. About 23GB."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/main/{CORPUS_JSONL}"
    request = urllib.request.Request(url, headers={"User-Agent": "research-agent/0.1"})
    hasher = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=600) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            hasher.update(chunk)
    return hasher.hexdigest()


def stream_pubmed_corpus(
    src: Path,
    dest: Path,
    *,
    gold_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the 16M dump into corpus.jsonl without loading it all into RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    gold_documents = gold_documents or []
    required = {str(doc["doc_id"]) for doc in gold_documents}
    found: set[str] = set()
    n_written = 0
    with src.open("r", encoding="utf-8") as incoming, dest.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            line = line.strip()
            if not line:
                continue
            rec = convert_pubmed_record(json.loads(line))
            if rec is None:
                continue
            outgoing.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if rec["doc_id"] in required:
                found.add(rec["doc_id"])
        n_gold_in_dump = len(found)
        n_appended = 0
        for doc in gold_documents:
            if doc["doc_id"] in found:
                continue
            outgoing.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_written += 1
            n_appended += 1
            found.add(doc["doc_id"])
    return {
        "n_documents": n_written,
        "n_gold": len(required),
        "n_gold_in_dump": n_gold_in_dump,
        "n_appended_gold": n_appended,
        "source": str(src),
        "output": str(dest),
    }
