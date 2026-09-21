from research_agent.contracts import TaskInput
from research_agent.evaluation.snapshot import redact_mapping, write_run_snapshot

from tests.support import TempDirTestCase


class TestSnapshot(TempDirTestCase):
    def test_redact_mapping_strips_secret_keys(self):
        cleaned = redact_mapping({"api_key": "abc", "model": "qwen", "nested": {"hf_token": "x"}})
        assert cleaned["api_key"] == "<redacted>"
        assert cleaned["nested"]["hf_token"] == "<redacted>"
        assert cleaned["model"] == "qwen"

    def test_write_run_snapshot_records_frozen_ids(self):
        tasks = [
            TaskInput(request_id="a", question="q", environment_id="e", task_id="a", split="train"),
        ]
        metrics = {"answer_em": 0.5, "unrun": False}
        resolved = write_run_snapshot(
            self.tmp_path,
            run_id="frozen-mini",
            metrics=metrics,
            tasks=tasks,
            eval_cfg={"frozen_task_ids": ["a"], "frozen": True},
            cli={"api_key": "should-not-leak", "split": "train"},
        )
        assert (self.tmp_path / "resolved_config.json").exists()
        assert resolved["frozen_task_ids"] == ["a"]
        assert resolved["cli"]["api_key"] == "<redacted>"
        written = (self.tmp_path / "metrics.json").read_text(encoding="utf-8")
        assert "frozen_task_ids" in written
        assert "should-not-leak" not in (self.tmp_path / "resolved_config.json").read_text(encoding="utf-8")
