from research_agent.evaluation.planning import (
    PLANNER_VERSION,
    compare_planning_conditions,
    frozen_plan,
    write_planning_review,
)

from tests.support import TempDirTestCase


class TestPlanning(TempDirTestCase):
    def test_frozen_plan_flags_cs005_markers(self):
        oracle = frozen_plan(
            {
                "answer": "yes: ExtraMix-2M and BoostSplit",
                "citations": ["cs:boostnet:1"],
                "claims": [{"text": "gain depends on ExtraMix-2M", "citation_ids": ["cs:boostnet:1"]}],
                "conditions": [],
                "unresolved_questions": [],
                "usage": {"explore_calls": 2, "completion_tokens": 40, "latency_ms": 12},
            }
        )
        assert oracle["planner_version"] == PLANNER_VERSION
        assert oracle["structural_flags"]["mentions_extramix"] is True
        assert oracle["structural_flags"]["mentions_boostsplit"] is True
        assert oracle["structural_flags"]["has_citations"] is True
        assert oracle["planning_cost"]["planner_tokens"] == 0
        unknown = frozen_plan({"answer": "unknown", "citations": [], "claims": [], "conditions": [], "unresolved_questions": []})
        assert unknown["structural_flags"]["mentions_extramix"] is False
        assert unknown["structural_flags"]["has_citations"] is False
        assert any("no citations" in item for item in unknown["missing_conditions"])

    def test_compare_planning_keeps_trained_unrun_and_empty_human_scores(self):
        oracle = self.tmp_path / "cpu-oracle"
        none = self.tmp_path / "synth-no-retrieval"
        oracle.mkdir()
        none.mkdir()
        (oracle / "episodes.jsonl").write_text(
            '{"episode_id":"o","task":{"task_id":"cs-005","question":"BoostNet gain?"},"result":{"answer":"yes: ExtraMix-2M and BoostSplit","citations":["cs:boostnet:1"],"policy_version":"scripted-oracle"}}\n',
            encoding="utf-8",
        )
        (none / "episodes.jsonl").write_text(
            '{"episode_id":"n","task":{"task_id":"cs-005","question":"BoostNet gain?"},"result":{"answer":"unknown","citations":[],"policy_version":"scripted-no-retrieval"}}\n',
            encoding="utf-8",
        )
        report = compare_planning_conditions(task_id="cs-005", outputs=self.tmp_path)
        by_name = {item["name"]: item for item in report["conditions"]}
        assert by_name["trained_agent"]["status"] == "not_run"
        assert by_name["trained_agent"]["human_scores"] is None
        assert by_name["scripted_oracle"]["status"] == "ran"
        assert by_name["scripted_oracle"]["human_scores"] is None
        assert by_name["scripted_oracle"]["plan"]["structural_flags"]["mentions_extramix"] is True
        assert by_name["no_retrieval"]["plan"]["structural_flags"]["mentions_extramix"] is False
        assert "not measured" in report["note"].lower()
        dest = self.tmp_path / "planning-review.json"
        written = write_planning_review(self.tmp_path, dest)
        assert dest.exists()
        assert written["trained_agent_status"] == "not_run"
