"""Load locally cached abstracts. Does not download from NCBI or Europe PMC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def fetch_abstracts(
    pmids: list[str],
    *,
    getter: Callable[[str], dict[str, str] | None] | None = None,
    cache_path: Path | None = None,
    **_kwargs,
) -> dict[str, dict[str, str]]:
    unique = []
    seen: set[str] = set()
    for pmid in pmids:
        if pmid and pmid not in seen:
            unique.append(pmid)
            seen.add(pmid)
    cache_path = cache_path or Path("data/raw/papersearchqa/abstracts.json")
    found: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        for pmid, item in json.loads(cache_path.read_text(encoding="utf-8")).items():
            if item and item.get("abstract"):
                found[pmid] = item
    if getter is not None:
        for pmid in unique:
            if pmid in found:
                continue
            try:
                item = getter(pmid)
            except Exception:
                item = None
            if item and item.get("abstract"):
                found[pmid] = item
    return {pmid: found[pmid] for pmid in unique if pmid in found}
