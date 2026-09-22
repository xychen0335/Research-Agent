from research_agent.models.factory import load_policy
from research_agent.training.sft import saved_adapter_manifest

from tests.support import TempDirTestCase


class TestAdapterReload(TempDirTestCase):
    def test_saved_adapter_manifest_requires_config_and_weights(self):
        empty = self.tmp_path / "missing"
        assert saved_adapter_manifest(empty)["reloadable"] is False
        dest = self.tmp_path / "adapter"
        dest.mkdir()
        (dest / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert saved_adapter_manifest(dest)["reloadable"] is False
        (dest / "adapter_model.safetensors").write_bytes(b"x")
        info = saved_adapter_manifest(dest)
        assert info["reloadable"] is True
        assert info["has_config"] is True
        assert info["has_weights"] is True

    def test_grpo_policy_requests_trainable_adapter(self):
        policy = load_policy(
            "huggingface",
            model_name="models/Qwen3.5-4B",
            adapter=None,
            local_files_only=True,
            trainable_adapter=True,
        )
        assert policy._trainable_adapter is True
        assert policy._adapter is None
        assert policy._local_files_only is True
        eval_policy = load_policy(
            "huggingface",
            model_name="models/Qwen3.5-4B",
            adapter="outputs/grpo/adapter",
            local_files_only=True,
        )
        assert eval_policy._trainable_adapter is False
