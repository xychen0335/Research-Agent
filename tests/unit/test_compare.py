import json

from research_agent.cli import cmd_eval_trained
from research_agent.evaluation.compare import compare_frozen_runs, write_compare_report

from tests.support import TempDirTestCase


class TestCompare(TempDirTestCase):
    def test_compare_does_not_zero_fill_missing(self):
        ran = self.tmp_path / "base"
        ran.mkdir()
        (ran / "metrics.json").write_text(
            json.dumps({"answer_em": 0.5, "n": 8, "policy_version": "qwen3.5:4b", "harness_version": "h1"}) + "\n",
            encoding="utf-8",
        )
        missing = self.tmp_path / "sft"
        missing.mkdir()
        report = compare_frozen_runs([ran, missing])
        assert report["runs"][0]["status"] == "ran"
        assert report["runs"][0]["answer_em"] == 0.5
        assert report["runs"][1]["status"] == "not_run"
        assert report["runs"][1]["answer_em"] is None
        out = self.tmp_path / "compare.json"
        write_compare_report([ran, missing], out)
        assert json.loads(out.read_text(encoding="utf-8"))["runs"][1]["answer_em"] is None

    def test_eval_trained_refuses_missing_adapter(self):
        class Args:
            adapter = str(self.tmp_path / "no-adapter")
            output = str(self.tmp_path / "frozen-sft")

        code = cmd_eval_trained(Args())
        assert code == 2
        report = json.loads((self.tmp_path / "frozen-sft" / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "not_run"
        assert not (self.tmp_path / "frozen-sft" / "metrics.json").exists()
