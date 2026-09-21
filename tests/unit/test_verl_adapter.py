from types import SimpleNamespace

from research_agent.contracts import TOOL_SEARCH, Budget, TaskInput
from research_agent.grading.contracts import GradingSpec
from research_agent.harness.loop import run_episode
from research_agent.models.scripted import ScriptedPolicy, tool_call
from research_agent.training.export import export_grpo_group, export_verl_row
from research_agent.training.verl.agent_loop import record_to_rollout
from research_agent.training.verl.reward_adapter import compute_score, submitted_answer

from tests.support import HarnessAsyncTestCase


class TestVerlAdapter(HarnessAsyncTestCase):
    async def test_response_mask_is_one_on_generations_zero_on_obs(self):
        task = TaskInput(
            request_id="bio-001",
            task_id="bio-001",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="synthetic-dev",
            budget=Budget(),
        )
        model = ScriptedPolicy(
            {
                task.question: [
                    tool_call(TOOL_SEARCH, {"query": "RB1"}),
                    tool_call("submit", {"answer": "RB1", "citations": []}),
                ]
            }
        )
        record = await run_episode(task, model, self.tools)
        trace = record.token_trace
        assert trace.usable_for_rl
        assert 1 in trace.response_mask
        assert 0 in trace.response_mask
        assert len(trace.response_ids) == len(trace.response_mask)
        rollout = record_to_rollout(record, reward_score=1.0)
        payload = rollout.as_agent_loop_dict()
        assert payload["prompt_ids"]
        assert payload["metrics"]["compute_score"] == 0.0
        assert payload["extra_fields"]["mask_convention"] == "1=model_token,0=observation"
        assert payload["extra_fields"]["usable_for_rl"] is True
        assert rollout.extra_fields["usable_for_rl"] is True

    async def test_grpo_group_flags_zero_variance(self):
        task = TaskInput(
            request_id="bio-001",
            task_id="bio-001",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="synthetic-dev",
            budget=Budget(),
        )
        model = ScriptedPolicy(
            {task.question: [tool_call("submit", {"answer": "RB1", "citations": []})]},
            policy_version="same",
        )
        records = [await run_episode(task, model, self.tools) for _ in range(4)]
        group = export_grpo_group(records, [0.0, 0.0, 0.0, 0.0], group_id="g1")
        assert group["all_fail"]
        assert group["zero_advantage_variance"]

    def test_task_from_chat_prompt(self):
        from research_agent.training.verl.agent_loop import task_from_dataset_row

        task = task_from_dataset_row(
            {
                "task_id": "psqa-1",
                "prompt": [{"role": "user", "content": "Which gene?"}],
                "extra_info": {"environment_id": "papersearchqa-dev"},
            }
        )
        assert task.question == "Which gene?"
        assert task.environment_id == "papersearchqa-dev"
        assert task.research_context == {}

    def test_verl_agent_loop_is_module_level(self):
        from research_agent.training.verl.agent_loop import (
            AGENT_LOOP_OUTPUT_FIELDS,
            VerlResearchAgentLoop,
            hydra_agent_loop_target,
            verl_agent_loop_class,
        )

        assert verl_agent_loop_class() is VerlResearchAgentLoop
        assert VerlResearchAgentLoop.__qualname__ == "VerlResearchAgentLoop"
        assert "<locals>" not in VerlResearchAgentLoop.__qualname__
        assert hydra_agent_loop_target()["_target_"].endswith("VerlResearchAgentLoop")
        assert VerlResearchAgentLoop.agent_name == "research_agent"
        import yaml
        from pathlib import Path

        listed = yaml.safe_load(Path("configs/training/verl_agent_loop.yaml").read_text(encoding="utf-8"))
        assert listed[0]["name"] == "research_agent"
        assert listed[0]["_target_"].endswith("VerlResearchAgentLoop")
        assert "prompt_ids" in AGENT_LOOP_OUTPUT_FIELDS
        assert "response_mask" in AGENT_LOOP_OUTPUT_FIELDS

    def test_compute_score_matches_verl_signature(self):
        score = compute_score(
            "research-agent",
            "RB1",
            "RB1",
            extra_info={"task_id": "bio-001", "aliases": ["Rb1"], "scoring": "exact_match"},
        )
        assert score == 1.0
        xml = tool_call("submit", {"answer": "wrong", "citations": []})
        assert submitted_answer(xml) == "wrong"
        assert compute_score("research-agent", xml, "RB1", extra_info={"scoring": "exact_match"}) == 0.0

    def test_export_verl_row_keeps_gold_out_of_prompt(self):
        task = TaskInput(
            request_id="bio-001",
            task_id="bio-001",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="synthetic-dev",
            split="train",
        )
        spec = GradingSpec(task_id="bio-001", answer="RB1", aliases=("Rb1",))
        row = export_verl_row(task, spec)
        assert row["prompt"] == [{"role": "user", "content": task.question}]
        assert "RB1" not in str(row["prompt"])
        assert row["agent_name"] == "research_agent"
        assert row["reward_model"]["ground_truth"] == "RB1"
        assert "Rb1" in row["extra_info"]["aliases"]

    def test_token_policy_uses_tokenizer_not_hash(self):
        from research_agent.training.verl.model_backend import VerlTokenPolicy

        class FakeTok:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
                assert tokenize is True
                return [11, 22, 33]

            def encode(self, text, add_special_tokens=False):
                return [7, 8]

            def decode(self, ids, skip_special_tokens=True):
                return "decoded"

        policy = VerlTokenPolicy(policy_version="tok", tokenizer=FakeTok())
        assert policy.encode_messages([{"role": "user", "content": "q"}]) == [11, 22, 33]
        assert policy.encode_text("hello") == [7, 8]
        assert policy.decode_ids([1]) == "decoded"

    async def test_rollout_from_kwargs_uses_token_backend(self):
        from research_agent.models.base import encode_text
        from research_agent.models.scripted import tool_call as make_call
        from research_agent.training.verl.agent_loop import AGENT_LOOP_OUTPUT_FIELDS, rollout_from_kwargs
        from research_agent.training.verl.model_backend import VerlTokenPolicy

        text = make_call("submit", {"answer": "RB1", "citations": []})
        token_ids = encode_text(text)

        async def generate_fn(prompt_ids, params):
            assert params.get("temperature") == 0.0
            return {
                "token_ids": token_ids,
                "logprobs": [-0.1] * len(token_ids),
                "text": text,
                "finish_reason": "stop",
            }

        backend = VerlTokenPolicy(generate_fn, policy_version="test-verl", sampling_params={"temperature": 1.0})
        rollout = await rollout_from_kwargs(
            tools=self.tools,
            backend=backend,
            sampling_params={"temperature": 0.0},
            kwargs={
                "task_id": "bio-001",
                "question": "Which gene is mutated in childhood retinoblastoma?",
                "environment_id": "synthetic-dev",
                "reward_model": {"ground_truth": "RB1"},
                "extra_info": {"scoring": "exact_match"},
            },
        )
        payload = rollout.as_agent_loop_dict()
        for key in AGENT_LOOP_OUTPUT_FIELDS:
            assert key in payload
        assert payload["reward_score"] == 1.0
        assert payload["extra_fields"]["policy_version"] == "test-verl"

    async def test_verl_research_agent_loop_run_returns_agent_loop_output(self):
        from research_agent.models.base import encode_text
        from research_agent.models.scripted import tool_call as make_call
        from research_agent.training.verl.agent_loop import VerlResearchAgentLoop
        from research_agent.training.verl.model_backend import VerlTokenPolicy

        text = make_call("submit", {"answer": "RB1", "citations": []})
        token_ids = encode_text(text)

        async def generate_fn(prompt_ids, params):
            return {"token_ids": token_ids, "logprobs": [-0.1] * len(token_ids), "text": text, "finish_reason": "stop"}

        loop = VerlResearchAgentLoop()
        loop._research_tools = self.tools
        loop._research_backend = VerlTokenPolicy(generate_fn, policy_version="loop-test")
        output = await loop.run(
            {"temperature": 0.0},
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="synthetic-dev",
            extra_info={"task_id": "bio-001"},
            reward_model={"ground_truth": "RB1"},
        )
        assert output.prompt_ids
        assert output.response_ids
        assert output.response_mask
        assert output.reward_score == 1.0
        assert output.extra_fields["mask_convention"] == "1=model_token,0=observation"

    async def test_backend_from_server_manager_forwards_logprobs(self):
        from research_agent.training.verl.agent_loop import backend_from_server_manager

        class FakeServer:
            async def generate(self, request_id, prompt_ids, sampling_params):
                return SimpleNamespace(token_ids=[7, 8], log_probs=[-0.2, -0.3], text="ok", finish_reason="stop")

        backend = backend_from_server_manager(FakeServer(), policy_version="verl-test")
        payload = await backend.generate_fn([1, 2], {"temperature": 0.2})
        assert payload["token_ids"] == [7, 8]
        assert payload["logprobs"] == [-0.2, -0.3]

    def test_verl_launch_overrides_point_at_framework_entry(self):
        from pathlib import Path

        from research_agent.training.verl.launch import hydra_overrides, load_overlay, verl_main_ppo_argv

        raw = load_overlay()
        assert raw["framework_entry"] == "verl.trainer.main_ppo"
        argv = verl_main_ppo_argv(root=Path.cwd())
        joined = " ".join(argv)
        assert "algorithm.adv_estimator=grpo" in argv
        assert "actor_rollout_ref.rollout.agent.default_agent_loop=research_agent" in argv
        assert "actor_rollout_ref.rollout.mode=async" in argv
        assert "data.return_raw_chat=True" in argv
        assert "actor_rollout_ref.rollout.n=4" in argv
        assert "reward.custom_reward_function.name=compute_score" in argv
        assert "research_agent.training.grpo" not in joined
        assert "Qwen/Qwen3.5-4B" in joined
        adapter_argv = verl_main_ppo_argv(root=Path.cwd(), adapter="/tmp/lora-adapter")
        assert any(item.startswith("+actor_rollout_ref.model.lora_adapter_path=") for item in adapter_argv)
        overrides = hydra_overrides(raw, root=Path.cwd())
        model_path = next(item for item in overrides if item.startswith("actor_rollout_ref.model.path="))
        assert model_path.endswith("Qwen/Qwen3.5-4B")

    def test_run_grpo_does_not_pretend_to_be_verl(self):
        from research_agent.training.grpo import GRPOConfig, run_grpo

        result = run_grpo(GRPOConfig(backend="verl", output_dir=str(self.tmp_path / "grpo")))
        assert result["status"] == "not_run"
        assert result["backend"] == "verl"
        assert "verl.trainer.main_ppo" in result["error"]
        assert (self.tmp_path / "grpo" / "grpo_report.json").exists()

    def test_run_verl_grpo_refuses_without_cuda(self):
        from research_agent.training.verl.launch import run_verl_grpo

        result = run_verl_grpo(output_dir=self.tmp_path / "verl-grpo", exec_process=False)
        assert result["status"] == "not_run"
        assert result["framework_entry"] == "verl.trainer.main_ppo"
        assert "error" in result
        assert (self.tmp_path / "verl-grpo" / "verl_grpo_report.json").exists()

    def test_write_verl_files_from_prepared(self):
        import sys
        from unittest.mock import patch

        from research_agent.training.export import write_verl_files

        # Parquet is optional. A broken local pyarrow import must not hang this test.
        with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
            report = write_verl_files(self.prepared.output_dir)
        assert report["status"] == "wrote"
        assert report["n_train"] + report["n_test"] == 32
        train_path = self.prepared.output_dir / "public" / "verl_train.jsonl"
        assert train_path.exists()
        first = train_path.read_text(encoding="utf-8").splitlines()[0]
        assert "research_agent" in first
        assert '"role": "user"' in first
