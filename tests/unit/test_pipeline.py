from pathlib import Path

from research_agent.training.pipeline import run_gpu_mini

from tests.support import TempDirTestCase


class TestPipeline(TempDirTestCase):
    def test_gpu_mini_refuses_without_device_or_weights(self):
        report = run_gpu_mini(
            grpo_config=Path("configs/training/grpo.yaml"),
            output_dir=self.tmp_path / "gpu-mini",
        )
        assert report["status"] == "not_run"
        assert report["sft"] is None
        assert "error" in report
        assert (self.tmp_path / "gpu-mini" / "report.json").exists()
        assert report.get("frozen_eval", {}).get("status") == "not_run"
