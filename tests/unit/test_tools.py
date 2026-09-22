from research_agent.contracts import TOOL_OPEN, TOOL_SEARCH, TOOL_SUBMIT, Action
from research_agent.harness.state import EpisodeState
from research_agent.contracts import TaskInput

from tests.support import HarnessAsyncTestCase


class TestTools(HarnessAsyncTestCase):
    async def test_search_returns_doc_ids(self):
        state = EpisodeState(
            task=TaskInput(request_id="t", question="q", environment_id="test"),
            episode_id="e",
            policy_version="p",
            harness_version="h",
            environment_version=self.tools.environment_version,
        )
        obs = await self.tools.execute(Action(TOOL_SEARCH, {"query": "childhood retinoblastoma RB1"}), state)
        ids = [hit["doc_id"] for hit in obs.content["hits"]]
        assert "bio:rb1" in ids
        blob = obs.as_text()
        assert "golden_answers" not in blob
        assert "gold_evidence" not in blob

    async def test_open_returns_stable_paragraph_ids(self):
        state = EpisodeState(
            task=TaskInput(request_id="t", question="q", environment_id="test"),
            episode_id="e",
            policy_version="p",
            harness_version="h",
            environment_version=self.tools.environment_version,
        )
        obs = await self.tools.execute(Action(TOOL_OPEN, {"doc_id": "bio:rb1", "start": 0, "end": 1}), state)
        ids = [item["paragraph_id"] for item in obs.content["paragraphs"]]
        assert ids == ["bio:rb1:0", "bio:rb1:1"]

    async def test_submit_rejects_unknown_paragraph(self):
        state = EpisodeState(
            task=TaskInput(request_id="t", question="q", environment_id="test"),
            episode_id="e",
            policy_version="p",
            harness_version="h",
            environment_version=self.tools.environment_version,
        )
        obs = await self.tools.execute(
            Action(TOOL_SUBMIT, {"answer": "RB1", "citations": ["missing:9"]}),
            state,
        )
        assert obs.error_kind == "invalid_params"

    async def test_empty_query_recorded(self):
        state = EpisodeState(
            task=TaskInput(request_id="t", question="q", environment_id="test"),
            episode_id="e",
            policy_version="p",
            harness_version="h",
            environment_version=self.tools.environment_version,
        )
        obs = await self.tools.execute(Action(TOOL_SEARCH, {"query": "  "}), state)
        assert obs.error_kind == "empty_result"
