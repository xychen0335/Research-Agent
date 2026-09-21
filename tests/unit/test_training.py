from research_agent.models.base import encode_text
from research_agent.models.factory import load_policy
from research_agent.models.openai_compatible import OpenAICompatiblePolicy
from research_agent.models.scripted import ScriptedPolicy, tool_call
from research_agent.training.grpo import clipped_surrogate, collect_group, group_advantages, sequence_grpo_loss
from research_agent.training.sft import IGNORE_INDEX, assistant_token_labels, load_sft_config

from tests.support import HarnessAsyncTestCase


class TestTraining(HarnessAsyncTestCase):
    def test_assistant_labels_ignore_user_tokens(self):
        messages = [
            {"role": "system", "content": "tools only"},
            {"role": "user", "content": "what gene"},
            {"role": "assistant", "content": "search now"},
        ]
        ids, labels = assistant_token_labels(messages, encode_text)
        assert len(ids) == len(labels)
        assert IGNORE_INDEX in labels
        assert any(item != IGNORE_INDEX for item in labels)
        assistant_ids = encode_text("assistant\nsearch now")
        assert labels[-len(assistant_ids) :] == assistant_ids

    def test_cached_hf_model_is_false_for_missing_qwen(self):
        from research_agent.training.compat import cached_hf_model

        assert cached_hf_model("Qwen/Qwen3.5-4B") is False
        assert cached_hf_model("/tmp/does-not-exist-qwen35") is False

    def test_sft_config_ignores_unknown_yaml_keys(self):
        path = self.tmp_path / "sft.yaml"
        path.write_text("method: lora_sft\nbase: Qwen/Qwen3.5-4B\nmax_steps: 3\nnote: x\n", encoding="utf-8")
        cfg = load_sft_config(path)
        assert cfg.base == "Qwen/Qwen3.5-4B"
        assert cfg.max_steps == 3
        assert cfg.min_teacher_rows == 16

    def test_select_frozen_ids_keeps_yaml_order(self):
        from research_agent.cli import _select_tasks
        from research_agent.contracts import TaskInput

        tasks = [
            TaskInput(request_id="b", question="b", environment_id="e", task_id="b", split="train"),
            TaskInput(request_id="a", question="a", environment_id="e", task_id="a", split="train"),
            TaskInput(request_id="c", question="c", environment_id="e", task_id="c", split="test"),
        ]
        selected = _select_tasks(tasks, split="train", limit=None, frozen_ids=["a", "b"])
        assert [item.public_id() for item in selected] == ["a", "b"]

    def test_run_sft_refuses_sparse_teacher(self):
        from research_agent.training.sft import SFTConfig, run_sft

        teacher = self.tmp_path / "sft.jsonl"
        teacher.write_text(
            '{"messages":[{"role":"assistant","content":"x"}]}\n' * 3,
            encoding="utf-8",
        )
        result = run_sft(
            SFTConfig(teacher_jsonl=str(teacher), output_dir=str(self.tmp_path / "out"), min_teacher_rows=16)
        )
        assert result["status"] == "not_run"
        assert result["teacher_rows"] == 3
        assert "only 3" in result["error"]

    def test_group_advantages_zero_mean(self):
        adv = group_advantages([1.0, 0.0, 1.0, 0.0])
        assert abs(sum(adv)) < 1e-9
        assert adv[0] > 0
        assert adv[1] < 0

    def test_clipped_surrogate_is_negative_for_positive_advantage(self):
        loss = clipped_surrogate(1.1, 1.0, 0.2)
        assert loss < 0

    def test_sequence_grpo_loss_masks_observations(self):
        loss = sequence_grpo_loss(
            logprobs=[0.0, -1.0],
            old_logprobs=[0.0, -1.0],
            mask=[0, 1],
            advantage=1.0,
        )
        assert loss < 0

    def test_load_policy_ollama_defaults(self):
        policy = load_policy("ollama")
        assert isinstance(policy, OpenAICompatiblePolicy)
        assert policy.model == "qwen3.5:4b"
        assert "11434" in policy.base_url
        assert policy.think is True
        assert policy.policy_version == "qwen3.5:4b"

    def test_message_text_merges_reasoning(self):
        from research_agent.models.openai_compatible import message_text

        text = message_text({"reasoning": "plan", "content": '<tool_call>\n{"name": "search"}'})
        assert "plan" in text
        assert text.rstrip().endswith("</tool_call>")

    def test_native_and_markdown_tools_become_xml(self):
        from research_agent.models.openai_compatible import message_text

        native = message_text(
            {"content": "", "tool_calls": [{"function": {"name": "search", "arguments": {"query": "RB1"}}}]}
        )
        assert '"name": "search"' in native
        md = message_text({"content": '```json\n{"tool": "open", "arguments": {"doc_id": "d"}}\n```'})
        assert '"name": "open"' in md

    def test_rewrite_ollama_history_uses_native_tool_roles(self):
        from research_agent.models.openai_compatible import rewrite_ollama_messages
        from research_agent.models.scripted import tool_call as make_call

        messages = rewrite_ollama_messages(
            [
                {"role": "system", "content": "<tool_call> crash bait"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": make_call("search", {"query": "RB1"})},
                {"role": "user", "content": "<tool_response>\n{\"tool\": \"search\"}\n</tool_response>"},
            ]
        )
        assert messages[0]["content"].startswith("You are a research retrieval agent")
        assert messages[2]["tool_calls"][0]["function"]["name"] == "search"
        assert messages[3]["role"] == "tool"

    def test_load_policy_ollama_keeps_explicit_model_name(self):
        policy = load_policy("ollama", model_name="qwen3.5:9b")
        assert policy.model == "qwen3.5:9b"
        policy = load_policy("scripted", scripts={"*": []})
        assert isinstance(policy, ScriptedPolicy)

    async def test_collect_group_flags_all_fail(self):
        tasks, specs, tools = self.prepared_stack
        task = next(item for item in tasks if item.public_id() == "bio-001")
        model = ScriptedPolicy(
            {task.question: [tool_call("submit", {"answer": "wrong", "citations": []})]},
            policy_version="grpo-probe",
        )
        group = await collect_group(task, specs[task.public_id()], model, tools, group_id="g-bio-001", group_size=4)
        assert group["n"] == 4
        assert group["all_fail"]
        assert group["zero_advantage_variance"]
        assert abs(sum(group["advantages"])) < 1e-9
