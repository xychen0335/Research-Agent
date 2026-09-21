import json

from fastapi.testclient import TestClient

from apps.dashboard.app import create_app

from tests.support import TempDirTestCase


class TestDashboard(TempDirTestCase):
    def test_unrun_run_is_not_zero_filled(self):
        outputs = self.tmp_path / "outputs"
        (outputs / "ghost").mkdir(parents=True)
        client = TestClient(create_app(outputs))
        payload = client.get("/api/runs").json()
        ghost = next(item for item in payload["runs"] if item["run_id"] == "ghost")
        assert ghost["unrun"] is True
        assert ghost["metrics"] is None
        assert ghost.get("answer_em") is None
        assert all(item.get("answer_em") is None for item in payload["runs"] if item["unrun"])

    def test_planned_grpo_is_unrun(self):
        client = TestClient(create_app(self.tmp_path / "outputs"))
        payload = client.get("/api/runs").json()
        ids = {item["run_id"] for item in payload["runs"]}
        assert "grpo-qwen35-4b" in ids
        assert "sft-qwen35-4b" not in ids
        grpo = client.get("/api/runs/grpo-qwen35-4b").json()
        assert grpo["unrun"] is True
        assert grpo["metrics"] is None
        assert grpo["answer_em"] is None
        assert "GRPO 未运行" in (grpo.get("note") or "")

    def test_index_serves_html(self):
        client = TestClient(create_app(self.tmp_path / "outputs"))
        response = client.get("/")
        assert response.status_code == 200
        assert "轨迹看板" in response.text

    def test_live_run_is_marked_live(self):
        client = TestClient(create_app(self.tmp_path / "outputs"))
        response = client.post("/api/live", json={"question": "Which gene is mutated in childhood retinoblastoma?", "run_id": "live"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["live"] is True
        assert payload["question"].startswith("Which gene")

    def test_preferred_compare_defaults(self):
        outputs = self.tmp_path / "outputs"
        (outputs / "psqa-qwen35-4b-base").mkdir(parents=True)
        (outputs / "psqa-no-retrieval").mkdir(parents=True)
        (outputs / "psqa-qwen35-4b-base" / "metrics.json").write_text('{"answer_em": 0.5, "unrun": false}\n', encoding="utf-8")
        (outputs / "psqa-no-retrieval" / "metrics.json").write_text('{"answer_em": 0.0, "unrun": false}\n', encoding="utf-8")
        client = TestClient(create_app(outputs))
        payload = client.get("/api/runs").json()
        assert payload["preferred_left"] == "psqa-qwen35-4b-base"
        assert payload["preferred_right"] == "psqa-no-retrieval"

    def test_showcase_cs005(self):
        outputs = self.tmp_path / "outputs"
        for name, answer in (("cpu-oracle", "extra data and BoostSplit"), ("synth-no-retrieval", "unknown")):
            dest = outputs / name
            dest.mkdir(parents=True)
            (dest / "metrics.json").write_text('{"answer_em": 1.0}\n', encoding="utf-8")
            (dest / "episodes.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": f"{name}-cs005",
                        "task": {
                            "task_id": "cs-005",
                            "question": "Does BoostNet's 4.2 point gain depend on extra data?",
                        },
                        "result": {"answer": answer, "citations": ["cs:boostnet:1"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        client = TestClient(create_app(outputs))
        payload = client.get("/api/showcase").json()
        assert payload["task_id"] == "cs-005"
        assert payload["left"]["answer"] == "extra data and BoostSplit"
        assert payload["right"]["answer"] == "unknown"
        assert "not a trained" in payload["note"]
        names = {item["name"]: item for item in payload["planning"]["conditions"]}
        assert names["trained_agent"]["status"] == "not_run"
        assert names["scripted_oracle"]["plan"]["structural_flags"]["mentions_boostsplit"] is True
        assert names["no_retrieval"]["plan"]["structural_flags"]["mentions_extramix"] is False
        assert names["scripted_oracle"]["human_scores"] is None
