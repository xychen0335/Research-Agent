"""GPU save/reload. Skips on this laptop; runs one LoRA step on a node with CUDA + cached 4B."""

from pathlib import Path

from research_agent.training.compat import cached_hf_model, probe
from research_agent.training.sft import SFTConfig, run_sft, saved_adapter_manifest

from tests.support import TempDirTestCase


class TestSaveReload(TempDirTestCase):
    def test_sft_save_reload_on_cuda(self):
        report = probe()
        if not report.get("cuda_available"):
            self.skipTest("CUDA not available; GPU update/save/reload is unverified")
        if not cached_hf_model("Qwen/Qwen3.5-4B"):
            self.skipTest("Qwen/Qwen3.5-4B is not on disk; refusing Hub download")
        rows = self.tmp_path / "sft.jsonl"
        rows.write_text(
            '{"messages":[{"role":"system","content":"tools"},{"role":"user","content":"q"},'
            '{"role":"assistant","content":"<tool_call>{\\"name\\":\\"submit\\",\\"arguments\\":{\\"answer\\":\\"RB1\\",\\"citations\\":[]}}</tool_call>"}],'
            '"split":"train"}\n',
            encoding="utf-8",
        )
        cfg = SFTConfig(
            base="Qwen/Qwen3.5-4B",
            jsonl=str(rows),
            output_dir=str(self.tmp_path / "sft"),
            max_steps=1,
            min_rows=1,
            max_seq_len=256,
        )
        result = run_sft(cfg)
        assert result.get("status") == "ran", result
        manifest = saved_adapter_manifest(Path(result["adapter"]))
        assert manifest["reloadable"]
        from research_agent.models.factory import load_policy

        policy = load_policy(
            "huggingface",
            model_name="Qwen/Qwen3.5-4B",
            adapter=result["adapter"],
            local_files_only=True,
            trainable_adapter=True,
        )
        policy._ensure_loaded()
        assert hasattr(policy._model, "peft_config")
        assert any(p.requires_grad for p in policy._model.parameters())
