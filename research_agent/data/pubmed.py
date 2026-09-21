"""PubMed / Europe PMC abstract fetch for development subsets. Not the 16M BioASQ dump."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

USER_AGENT = "research-agent/0.1 (PaperSearchQA subset; https://github.com/)"


def _get(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_europepmc(pmid: str) -> dict[str, str] | None:
    query = urllib.parse.quote(f"EXT_ID:{pmid} AND SRC:MED")
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&resultType=core&format=json"
    payload = json.loads(_get(url).decode("utf-8"))
    hits = ((payload.get("resultList") or {}).get("result") or [])
    if not hits:
        return None
    hit = hits[0]
    abstract = str(hit.get("abstractText") or "").strip()
    title = str(hit.get("title") or "").strip()
    if not abstract:
        return None
    return {"pmid": pmid, "title": title, "abstract": abstract, "source": "europepmc"}


def fetch_ncbi(pmids: list[str]) -> dict[str, dict[str, str]]:
    if not pmids:
        return {}
    ids = ",".join(pmids)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={ids}&rettype=abstract&retmode=xml&tool=research-agent&email=research-agent@localhost"
    )
    root = ET.fromstring(_get(url))
    found: dict[str, dict[str, str]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.findtext(".//PMID")
        title = (article.findtext(".//ArticleTitle") or "").strip()
        parts = [el.text or "" for el in article.findall(".//Abstract/AbstractText")]
        abstract = " ".join(part.strip() for part in parts if part.strip()).strip()
        if pmid_el and abstract:
            found[str(pmid_el)] = {
                "pmid": str(pmid_el),
                "title": title,
                "abstract": abstract,
                "source": "ncbi",
            }
    return found


def fetch_abstracts(
    pmids: list[str],
    *,
    sleep_s: float = 0.34,
    getter: Callable[[str], dict[str, str] | None] | None = None,
    cache_path: Path | None = None,
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
        found.update(json.loads(cache_path.read_text(encoding="utf-8")))
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
    missing = [pmid for pmid in unique if pmid not in found]
    for start in range(0, len(missing), 40):
        batch = missing[start : start + 40]
        try:
            found.update(fetch_ncbi(batch))
        except Exception:
            continue
        time.sleep(sleep_s)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    return {pmid: found[pmid] for pmid in unique if pmid in found}
