from research_agent.data.sources.synthetic_dev import TASKS
from research_agent.environment.retrieval import BM25Index

from tests.support import HarnessTestCase


class TestRetrieval(HarnessTestCase):
    def test_support_docs_are_retrievable(self):
        index = BM25Index(self.corpus)
        missed = []
        for task in TASKS:
            if not task.spec.answerable or not task.spec.support_doc_ids:
                continue
            hits = {hit.doc_id for hit in index.search(task.search_hint, topk=5)}
            if set(task.spec.support_doc_ids).isdisjoint(hits):
                missed.append(task.task_id)
        assert missed == []

    def test_question_retrieves_retinoblastoma_doc(self):
        index = BM25Index(self.corpus)
        hits = [hit.doc_id for hit in index.search("childhood retinoblastoma gene", topk=5)]
        assert "bio:rb1" in hits
