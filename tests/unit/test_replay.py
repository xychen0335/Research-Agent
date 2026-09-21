from research_agent.contracts import TOOL_SEARCH, TOOL_SUBMIT, Budget, TaskInput, to_plain
from research_agent.harness.loop import run_episode
from research_agent.harness.trajectory import replay_result
from research_agent.models.scripted import ScriptedPolicy, tool_call

from tests.support import HarnessAsyncTestCase


class TestReplay(HarnessAsyncTestCase):
    async def test_replay_matches_episode_end(self):
        task = TaskInput(
            request_id="bio-001",
            task_id="bio-001",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="synthetic-dev",
            budget=Budget(),
        )
        model = ScriptedPolicy(
            {
                task.question: [
                    tool_call(TOOL_SEARCH, {"query": "retinoblastoma"}),
                    tool_call("open", {"doc_id": "bio:rb1", "start": 0, "end": 0}),
                    tool_call(TOOL_SUBMIT, {"answer": "RB1", "citations": ["bio:rb1:0"]}),
                ]
            }
        )
        record = await run_episode(task, model, self.tools)
        replayed = replay_result([to_plain(event) for event in record.events])
        assert replayed["answer"] == "RB1"
        assert replayed["citations"] == ["bio:rb1:0"]
        assert replayed["status"] == record.result.status
