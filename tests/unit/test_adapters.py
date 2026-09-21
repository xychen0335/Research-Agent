from pathlib import Path

from research_agent.data.sources.papersearchqa import convert_papersearchqa_rows
from research_agent.data.sources.qasper import convert_qasper_rows
from research_agent.data.synthesize import tasks_from_facts
from research_agent.environment.corpus import CorpusSnapshot

from tests.support import TempDirTestCase


class TestAdapters(TempDirTestCase):
    def test_papersearchqa_splits_public_and_private(self):
        rows = [
            {
                "question": "What gene?",
                "answer": "RB1",
                "golden_answers": ["RB1", "Rb1"],
                "pmid": "1",
                "paper_title": "Retinoblastoma",
                "abstract": "RB1 is mutated in retinoblastoma.",
                "cat": "genetics",
            }
        ]
        tasks, grading, docs = convert_papersearchqa_rows(rows, split="train")
        assert "answer" not in tasks[0]
        assert grading[0]["answer"] == "RB1"
        assert docs[0]["doc_id"] == "pmid:1"

    def test_qasper_uses_f1_protocol(self):
        rows = [
            {
                "id": "p1",
                "title": "A paper",
                "abstract": "We use LoRA rank 16.",
                "qas": [
                    {
                        "question": "What rank?",
                        "question_id": "q1",
                        "answers": [{"free_form_answer": "16"}],
                    }
                ],
            }
        ]
        tasks, grading, _docs = convert_qasper_rows(rows, split="test")
        assert grading[0]["scoring"] == "f1"
        assert tasks[0]["category"] == "single_paper"

    def test_prepare_qasper_fixture(self):
        from research_agent.data.prepare import prepare_qasper

        prepared = prepare_qasper(
            Path("tests/fixtures/qasper_mini.json"),
            self.tmp_path / "qasper-mini",
            split="test",
        )
        assert prepared.n_tasks == 1
        assert prepared.report["validation"]["ok"] is True
        assert prepared.report["note"].startswith("Single-paper")

    def test_synthesize_requires_existing_paragraphs(self):
        corpus = CorpusSnapshot.from_records(
            [{"doc_id": "d", "title": "t", "paragraphs": ["fact lives here"]}]
        )
        tasks, grading = tasks_from_facts(
            [
                {
                    "question": "Where is the fact?",
                    "answer": "here",
                    "evidence_paragraph_ids": ["d:0"],
                }
            ],
            corpus,
        )
        assert tasks[0]["task_id"]
        assert grading[0]["gold_evidence_ids"] == ["d:0"]
