from research_agent.evaluation.baselines import run_baseline
from research_agent.evaluation.planning import export_planning_packets
from research_agent.grading.reward import score_episode

from tests.support import HarnessAsyncTestCase


class TestEpisode(HarnessAsyncTestCase):
    async def test_oracle_agent_recovers_retinoblastoma(self):
        tasks, specs, tools = self.prepared_stack
        subset = [task for task in tasks if task.public_id() == "bio-001"]
        result = await run_baseline(
            "agent",
            subset,
            specs,
            tools,
            run_id="oracle-bio",
            output_dir=self.tmp_path / "oracle-bio",
            oracle=True,
        )
        record = result.records[0]
        score = score_episode(record.result, specs["bio-001"], max_explore=6)
        assert record.result.answer == "RB1"
        assert score.answer_score == 1.0
        assert score.legal_citation_rate == 1.0
        assert record.live is True
        assert (self.tmp_path / "oracle-bio" / "events.jsonl").exists()

    async def test_no_retrieval_baseline_is_wrong_on_factoid(self):
        tasks, specs, tools = self.prepared_stack
        subset = [task for task in tasks if task.public_id() == "bio-001"]
        result = await run_baseline("no_retrieval", subset, specs, tools, run_id="no-ret")
        score = score_episode(result.records[0].result, specs["bio-001"], max_explore=6)
        assert score.answer_score == 0.0

    async def test_same_harness_for_eval_and_planning(self):
        tasks, specs, tools = self.prepared_stack
        subset = [task for task in tasks if task.public_id() == "bio-001"]
        result = await run_baseline("agent", subset, specs, tools, run_id="h1", oracle=True)
        messages = result.records[0].messages
        assert messages[0]["role"] == "system"
        assert any("<tool_call>" in item["content"] for item in messages if item["role"] == "assistant")
        packets = export_planning_packets(result.records)
        assert packets[0]["question"]
        assert packets[0]["plan"]["planner_version"]
        assert packets[0]["human_scores"] is None
        assert "Human blind review" in packets[0]["note"]

    async def test_prepare_validation_ok(self):
        assert self.prepared.n_tasks == 2
        assert self.prepared.report["validation"]["ok"] is True
