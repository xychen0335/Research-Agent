"""verl Agent Loop plugin.

verl is the training framework: PPOTrainer owns rollout/train, weight sync, and GRPO.
This module only implements the user hook (`AgentLoopBase.run`) by calling the shared
Research harness. Hydra instantiates `VerlResearchAgentLoop` from:

    actor_rollout_ref.rollout.agent.default_agent_loop=research_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/training/verl_agent_loop.yaml

The nested factory that previously hid the class from Hydra is gone. CPU tests use the
same class against a dummy `AgentLoopBase` when verl is not installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_agent.contracts import Budget, TaskInput
from research_agent.environment.tools import ToolEnvironment
from research_agent.harness.loop import run_episode
from research_agent.harness.trajectory import EpisodeRecord
from research_agent.training.verl.model_backend import VerlTokenPolicy
from research_agent.training.verl.reward_adapter import compute_score


try:
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )

    _VERL_AVAILABLE = True
except ImportError:  # CPU / docs machines do not install verl
    _VERL_AVAILABLE = False

    def register(agent_name: str):
        def decorator(subclass):
            subclass.agent_name = agent_name
            subclass._hydra_target = {"_target_": f"{subclass.__module__}.{subclass.__qualname__}"}
            return subclass

        return decorator

    class AgentLoopBase:  # noqa: D401 - stand-in matching verl's constructor kwargs
        def __init__(self, *args: Any, **kwargs: Any):
            trainer_config = kwargs.get("trainer_config")
            self.config = getattr(trainer_config, "config", trainer_config)
            self.server_manager = kwargs.get("server_manager")
            self.tokenizer = kwargs.get("tokenizer")
            self.processor = kwargs.get("processor")
            data_config = kwargs.get("data_config")
            inner = getattr(data_config, "config", data_config) if data_config is not None else {}
            self.apply_chat_template_kwargs = _cfg_get(inner, "apply_chat_template_kwargs") or {}

    @dataclass
    class AgentLoopMetrics:
        generate_sequences: float = 0.0
        tool_calls: float = 0.0
        compute_score: float = 0.0

    @dataclass
    class AgentLoopOutput:
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]
        metrics: AgentLoopMetrics
        response_logprobs: list[float] | None = None
        num_turns: int = 0
        reward_score: float | None = None
        extra_fields: dict[str, Any] = field(default_factory=dict)


_TOOLS_CACHE: dict[str, ToolEnvironment] = {}

AGENT_LOOP_OUTPUT_FIELDS = (
    "prompt_ids",
    "response_ids",
    "response_mask",
    "response_logprobs",
    "num_turns",
    "reward_score",
    "metrics",
    "extra_fields",
)


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    current = cfg
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
            continue
        getter = getattr(current, "get", None)
        if callable(getter):
            current = getter(key)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


@dataclass
class RolloutOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    num_turns: int
    reward_score: float | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)
    usable_for_rl: bool = False

    def as_agent_loop_dict(self) -> dict[str, Any]:
        return {
            "prompt_ids": self.prompt_ids,
            "response_ids": self.response_ids,
            "response_mask": self.response_mask,
            "response_logprobs": self.response_logprobs,
            "num_turns": self.num_turns,
            "reward_score": self.reward_score,
            "usable_for_rl": self.usable_for_rl,
            "metrics": {
                "generate_sequences": float(self.extra_fields.get("generate_ms") or 0.0),
                "tool_calls": float(self.extra_fields.get("tool_ms") or 0.0),
                "compute_score": float(self.extra_fields.get("score_ms") or 0.0),
            },
            "extra_fields": self.extra_fields,
        }


def record_to_rollout(record: EpisodeRecord, *, reward_score: float | None = None) -> RolloutOutput:
    trace = record.token_trace
    return RolloutOutput(
        prompt_ids=list(trace.prompt_ids),
        response_ids=list(trace.response_ids),
        response_mask=list(trace.response_mask),
        response_logprobs=list(trace.response_logprobs) if trace.response_logprobs is not None else None,
        num_turns=record.result.usage.generation_turns,
        reward_score=reward_score,
        extra_fields={
            "episode_id": record.episode_id,
            "task_id": record.task.public_id(),
            "termination": record.result.termination,
            "policy_version": record.result.policy_version,
            "harness_version": record.result.harness_version,
            "environment_version": record.result.environment_version,
            "usable_for_rl": trace.usable_for_rl,
            "mask_convention": "1=model_token,0=observation",
            "generate_ms": record.result.usage.latency_ms,
            "tool_ms": 0.0,
            "submitted_answer": record.result.answer,
            "citations": list(record.result.citations),
            "explore_calls": record.result.usage.explore_calls,
        },
        usable_for_rl=trace.usable_for_rl,
    )


class ResearchAgentLoop:
    """Same harness path used by the verl subclass; kept for CPU tests without Hydra."""

    def __init__(self, tools: ToolEnvironment, *, policy_version: str = "verl-rollout"):
        self.tools = tools
        self.policy_version = policy_version

    async def run_task(self, task: TaskInput, backend: VerlTokenPolicy, *, reward_score: float | None = None) -> RolloutOutput:
        record = await run_episode(task, backend, self.tools)
        return record_to_rollout(record, reward_score=reward_score)


def tools_from_data_dir(data_dir: Any) -> ToolEnvironment:
    from research_agent.evaluation.runner import snapshot_from_dir

    key = str(Path(data_dir).resolve())
    cached = _TOOLS_CACHE.get(key)
    if cached is not None:
        return cached
    _tasks, _specs, tools = snapshot_from_dir(Path(data_dir))
    _TOOLS_CACHE[key] = tools
    return tools


def attach_environment(loop: Any, data_dir: Any) -> None:
    loop._research_tools = tools_from_data_dir(data_dir)


def backend_from_server_manager(
    server_manager: Any,
    *,
    policy_version: str = "verl-rollout",
    sampling_params: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> VerlTokenPolicy:
    """Wrap verl `LLMServerClient.generate` so the shared harness stays token-in/token-out."""

    async def generate_fn(prompt_ids: list[int], params: dict[str, Any]) -> dict[str, Any]:
        output = await server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=params,
        )
        if isinstance(output, (list, tuple)):
            token_ids = list(output)
            logprobs = None
            text = ""
            finish_reason = "stop"
        else:
            token_ids = list(getattr(output, "token_ids", None) or getattr(output, "output_ids", None) or [])
            logprobs = getattr(output, "log_probs", None)
            if logprobs is None:
                logprobs = getattr(output, "logprobs", None)
            text = str(getattr(output, "text", "") or "")
            finish_reason = str(getattr(output, "finish_reason", None) or "stop")
        if not text and tokenizer is not None and token_ids:
            decode = getattr(tokenizer, "decode", None)
            if callable(decode):
                text = str(decode(token_ids, skip_special_tokens=True) or "")
        return {
            "token_ids": token_ids,
            "logprobs": list(logprobs) if logprobs is not None else None,
            "text": text,
            "finish_reason": finish_reason,
        }

    return VerlTokenPolicy(
        generate_fn,
        policy_version=policy_version,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )


def _question_from_row(row: dict[str, Any]) -> str:
    if row.get("question"):
        return str(row["question"])
    prompt = row.get("prompt") or row.get("raw_prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict) and item.get("role") == "user" and item.get("content"):
                return str(item["content"])
        texts = [str(item.get("content") or "") for item in prompt if isinstance(item, dict)]
        return next((text for text in texts if text), "")
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return str(extra.get("question") or "")


def extra_environment(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return str(extra.get("environment_id") or "default")


def task_from_dataset_row(row: dict[str, Any], budget: Budget | None = None) -> TaskInput:
    question = _question_from_row(row)
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return TaskInput(
        request_id=str(row.get("task_id") or extra.get("task_id") or row.get("request_id") or "row"),
        task_id=str(row.get("task_id") or extra.get("task_id") or row.get("request_id") or "row"),
        question=question,
        environment_id=str(row.get("environment_id") or extra_environment(row)),
        budget=budget or Budget(),
        research_context={},
        split=str(row.get("split") or extra.get("split") or "train"),
    )


def _reward_from_row(record: EpisodeRecord, row: dict[str, Any]) -> float | None:
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        ground_truth = extra.get("ground_truth")
    if ground_truth is None:
        return None
    payload = {
        **extra,
        "submitted_answer": record.result.answer,
        "citations": list(record.result.citations),
        "explore_calls": record.result.usage.explore_calls,
        "max_explore_calls": record.task.budget.max_explore_calls,
    }
    return compute_score(
        str(row.get("data_source") or "research-agent"),
        record.result.answer,
        ground_truth,
        extra_info=payload,
    )


async def rollout_from_kwargs(
    *,
    tools: ToolEnvironment,
    backend: VerlTokenPolicy,
    sampling_params: dict[str, Any] | None,
    kwargs: dict[str, Any],
    reward_score: float | None = None,
) -> RolloutOutput:
    if sampling_params and hasattr(backend, "sampling_params"):
        backend.sampling_params = {**getattr(backend, "sampling_params", {}), **sampling_params}
    task = task_from_dataset_row(kwargs)
    record = await run_episode(task, backend, tools)
    if reward_score is None:
        reward_score = _reward_from_row(record, kwargs)
    return record_to_rollout(record, reward_score=reward_score)


def _data_dir_from_loop(loop: Any) -> str | None:
    env_dir = os.environ.get("RESEARCH_AGENT_DATA")
    if env_dir:
        return env_dir
    cfg = getattr(loop, "config", None)
    for keys in (
        ("research_agent", "data"),
        ("actor_rollout_ref", "rollout", "agent", "data"),
    ):
        value = _cfg_get(cfg, *keys)
        if value:
            return str(value)
    default = Path("data/processed/papersearchqa")
    if (default / "public" / "tasks.jsonl").exists():
        return str(default)
    return None


def hydra_agent_loop_target() -> dict[str, str]:
    return {"_target_": f"{VerlResearchAgentLoop.__module__}.{VerlResearchAgentLoop.__qualname__}"}


def verl_agent_loop_class():
    """Return the module-level Hydra-instantiable class. Kept for older imports."""
    return VerlResearchAgentLoop


@register("research_agent")
class VerlResearchAgentLoop(AgentLoopBase):
    """User-defined Agent Loop. verl's trainer calls `run`; this class does not update weights."""

    agent_name = "research_agent"

    def __init__(self, *args: Any, data_dir: str | None = None, **kwargs: Any):
        kwargs.pop("tools", None)
        super().__init__(*args, **kwargs)
        resolved = data_dir or _data_dir_from_loop(self)
        if resolved and (Path(resolved) / "public" / "tasks.jsonl").exists():
            attach_environment(self, resolved)

    def _backend(self, sampling_params: dict[str, Any]) -> VerlTokenPolicy:
        attached = getattr(self, "_research_backend", None)
        if attached is not None:
            if sampling_params:
                attached.sampling_params = {**getattr(attached, "sampling_params", {}), **sampling_params}
            return attached
        server = getattr(self, "server_manager", None)
        if server is None:
            raise RuntimeError(
                "VerlResearchAgentLoop needs verl's server_manager. This class is an Agent Loop "
                "plugin, not a trainer; start GRPO with scripts/train/verl_grpo.sh."
            )
        backend = backend_from_server_manager(
            server,
            policy_version=getattr(self, "policy_version", "verl-rollout"),
            sampling_params=sampling_params,
            tokenizer=getattr(self, "tokenizer", None),
            chat_template_kwargs=getattr(self, "apply_chat_template_kwargs", None) or {},
        )
        self._research_backend = backend
        return backend

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        tools = kwargs.pop("tools", None) or getattr(self, "_research_tools", None)
        backend = kwargs.pop("backend", None) or self._backend(sampling_params)
        if tools is None:
            raise RuntimeError(
                "Attach ToolEnvironment via attach_environment() or RESEARCH_AGENT_DATA / "
                "research_agent.data. The Agent Loop does not ship a retrieval corpus."
            )
        rollout = await rollout_from_kwargs(
            tools=tools,
            backend=backend,
            sampling_params=sampling_params,
            kwargs=kwargs,
        )
        payload = rollout.as_agent_loop_dict()
        metrics_raw = payload.get("metrics") or {}
        return AgentLoopOutput(
            prompt_ids=payload["prompt_ids"],
            response_ids=payload["response_ids"],
            response_mask=payload["response_mask"],
            response_logprobs=payload.get("response_logprobs"),
            num_turns=int(payload.get("num_turns") or 0),
            reward_score=payload.get("reward_score"),
            metrics=AgentLoopMetrics(
                generate_sequences=float(metrics_raw.get("generate_sequences") or 0.0),
                tool_calls=float(metrics_raw.get("tool_calls") or 0.0),
                compute_score=float(metrics_raw.get("compute_score") or 0.0),
            ),
            extra_fields=payload.get("extra_fields") or {},
        )


VerlResearchAgentLoop.AgentLoopOutput = AgentLoopOutput
