from pathlib import Path
from unittest.mock import patch

from research_agent.data.prepare import prepare_papersearchqa_full, prepare_source
from research_agent.data.sources.papersearchqa import convert_pubmed_record, stream_pubmed_corpus
from research_agent.data.validate import validate_prepared
from research_agent.environment.corpus import CorpusSnapshot

from tests.support import TempDirTestCase


def _qa_row(split: str, index: int, pmid: str) -> dict:
    return {
        "question": f"question {split} {index}",
        "answer": "RB1",
        "golden_answers": ["RB1"],
        "pmid": pmid,
        "paper_title": f"paper {pmid}",
        "cat": "genetics",
    }


class TestPapersearchqaFull(TempDirTestCase):
    def test_convert_pubmed_record_accepts_search_r1_contents(self):
        rec = convert_pubmed_record({"id": "99", "contents": "Title here. Abstract body."})
        assert rec["doc_id"] == "pmid:99"
        assert rec["title"] == "Title here"
        assert rec["paragraphs"] == ["Abstract body."]

    def test_convert_pubmed_record_accepts_title_abstract(self):
        rec = convert_pubmed_record({"pmid": "pmid:12", "title": "T", "abstract": "A"})
        assert rec["doc_id"] == "pmid:12"
        assert rec["paragraphs"] == ["A"]

    def test_stream_pubmed_appends_missing_gold_docs(self):
        src = self.tmp_path / "pubmed.jsonl"
        src.write_text(
            '{"pmid": "1", "title": "Gold", "abstract": "RB1 abstract."}\n',
            encoding="utf-8",
        )
        dest = self.tmp_path / "corpus.jsonl"
        gold = [
            {
                "doc_id": "pmid:1",
                "title": "Gold",
                "paragraphs": ["RB1 abstract."],
                "source": "pmid:1",
                "version": "v",
                "metadata": {"pmid": "1"},
            },
            {
                "doc_id": "pmid:2",
                "title": "Missing from dump",
                "paragraphs": ["filled from NCBI"],
                "source": "pmid:2",
                "version": "v",
                "metadata": {"pmid": "2"},
            },
        ]
        report = stream_pubmed_corpus(src, dest, gold_documents=gold)
        lines = dest.read_text(encoding="utf-8").splitlines()
        assert report["n_documents"] == 2
        assert report["n_appended_gold"] == 1
        assert any('"doc_id": "pmid:2"' in line for line in lines)

    def test_prepare_full_writes_official_splits(self):
        def fake_load(split, indices, raw_dir):
            assert indices is None
            if split == "train":
                rows = [_qa_row("train", i, f"t{i}") for i in range(3)]
            else:
                rows = [_qa_row("test", i, f"e{i}") for i in range(2)]
            return rows, {"loader": "mock", "sha256": "x", "path": "mock"}

        def fake_fetch(pmids, getter=None, **kwargs):
            return {
                pmid: {"pmid": pmid, "title": f"t {pmid}", "abstract": f"RB1 in {pmid}.", "source": "mock"}
                for pmid in pmids
            }

        with (
            patch("research_agent.data.sources.papersearchqa.load_split_rows", fake_load),
            patch("research_agent.data.pubmed.fetch_abstracts", fake_fetch),
        ):
            prepared = prepare_papersearchqa_full(self.tmp_path / "psqa")
        assert prepared.source == "papersearchqa"
        assert prepared.report["n_train"] == 3
        assert prepared.report["n_test"] == 2
        assert prepared.n_tasks == 5
        assert prepared.report["validation"]["ok"] is True

    def test_prepare_source_papersearchqa_does_not_need_input(self):
        with patch("research_agent.data.prepare.prepare_papersearchqa_full") as mocked:
            mocked.return_value = object()
            prepare_source("papersearchqa", self.tmp_path / "out")
        mocked.assert_called_once()

    def test_large_corpus_skips_bm25_audit_but_checks_support_docs(self):
        tasks = [{"task_id": "t1", "question": "q", "split": "train"}]
        grading = [
            {
                "task_id": "t1",
                "answer": "RB1",
                "support_doc_ids": ["d"],
                "gold_evidence_ids": ["d:0"],
                "answerable": True,
            }
        ]
        corpus = CorpusSnapshot.from_records([{"doc_id": "d", "title": "t", "paragraphs": ["RB1"]}])
        report = validate_prepared(tasks, grading, corpus, retrieval_audit=False)
        assert report.as_dict()["ok"] is True
        assert report.unretrievable == []
        assert any("skipped" in item for item in report.warnings)
