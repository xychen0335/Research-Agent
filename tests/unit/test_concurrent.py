from research_agent.contracts import TOOL_SEARCH, Budget, TaskInput
from research_agent.harness.loop import run_batch
from research_agent.models.scripted import ScriptedPolicy, tool_call

from tests.support import HarnessAsyncTestCase


class TestConcurrent(HarnessAsyncTestCase):
    async def test_concurrent_episodes_do_not_share_opened_paragraphs(self):
        t1 = TaskInput(
            request_id="bio-001",
            task_id="bio-001",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="test",
            budget=Budget(),
        )
        t2 = TaskInput(
            request_id="bio-002",
            task_id="bio-002",
            question="Which hormone is secreted by pancreatic beta cells?",
            environment_id="test",
            budget=Budget(),
        )
        model = ScriptedPolicy(
            {
                t1.question: [
                    tool_call(TOOL_SEARCH, {"query": "retinoblastoma"}),
                    tool_call("open", {"doc_id": "bio:rb1", "start": 0, "end": 0}),
                    tool_call("submit", {"answer": "RB1", "citations": ["bio:rb1:0"]}),
                ],
                t2.question: [
                    tool_call(TOOL_SEARCH, {"query": "pancreatic beta cells insulin"}),
                    tool_call("open", {"doc_id": "bio:other", "start": 0, "end": 0}),
                    tool_call("submit", {"answer": "insulin", "citations": ["bio:other:0"]}),
                ],
            }
        )
        records = await run_batch([t1, t2], model, self.tools, max_concurrency=2)
        by_id = {record.task.public_id(): record for record in records}
        assert by_id["bio-001"].result.citations == ["bio:rb1:0"]
        assert by_id["bio-002"].result.citations == ["bio:other:0"]
        assert by_id["bio-001"].result.unread_citations == []
        assert by_id["bio-002"].result.unread_citations == []
