"""GRPO group collection and a single-card HuggingFace fallback.

verl remains the GPU training framework. This module:
- collects same-question groups through the shared harness
- computes group-normalized advantages
- runs a clipped surrogate update when a HuggingFace LoRA policy is attached

It does not launch or reimplement verl. Use `scripts/train/verl_grpo.sh`
(`python -m verl.trainer.main_ppo`) for the framework path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from research_agent.contracts import TaskInput
from research_agent.grading.contracts import GradingSpec
from research_agent.grading.reward import score_episode
from research_agent.harness.loop import PolicyModel, ToolDispatcher, run_episode
from research_agent.harness.trajectory import EpisodeRecord, dump_jsonl
from research_agent.training.compat import cached_hf_model, probe, write_report
from research_agent.training.export import export_grpo_group


@dataclass
class GRPOConfig:
    group_size: int = 4
    context_tokens: int = 8192
    cost_lambda: float = 0.0
    clip: float = 0.2
    kl_coef: float = 0.02
    learning_rate: float = 1.0e-6
    max_prompts: int = 4
    max_updates: int = 1
    data: str = "data/processed/papersearchqa"
    output_dir: str = "outputs/grpo"
    policy: str = "huggingface"
    model_name: str = "Qwen/Qwen3.5-4B"
    adapter: str = ""
    backend: str = "auto"


def load_grpo_config(path: Path | None) -> GRPOConfig:
    raw: dict[str, Any] = {}
    if path and path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            raw = loaded
    fields = {key: raw[key] for key in GRPOConfig.__dataclass_fields__ if key in raw}
    return GRPOConfig(**fields)


def group_advantages(rewards: Sequence[float]) -> list[float]:
    if not rewards:
        return []
    mean_r = sum(rewards) / len(rewards)
    return [float(reward) - mean_r for reward in rewards]


def clipped_surrogate(ratio: float, advantage: float, clip: float = 0.2) -> float:
    clipped = max(1.0 - clip, min(1.0 + clip, ratio))
    return -min(ratio * advantage, clipped * advantage)


def sequence_grpo_loss(
    logprobs: Sequence[float],
    old_logprobs: Sequence[float],
    mask: Sequence[int],
    advantage: float,
    *,
    clip: float = 0.2,
    ref_logprobs: Sequence[float] | None = None,
    kl_coef: float = 0.0,
) -> float:
    if not logprobs or not old_logprobs:
        return 0.0
    n = 0
    total = 0.0
    kl_term = 0.0
    limit = min(len(logprobs), len(old_logprobs), len(mask))
    for i in range(limit):
        if mask[i] != 1:
            continue
        ratio = math.exp(logprobs[i] - old_logprobs[i])
        total += clipped_surrogate(ratio, advantage, clip)
        n += 1
        if ref_logprobs is not None and i < len(ref_logprobs) and kl_coef:
            kl_term += (logprobs[i] - ref_logprobs[i])
    if n == 0:
        return 0.0
    loss = total / n
    if ref_logprobs is not None and kl_coef:
        loss += kl_coef * (kl_term / n)
    return loss


def tensor_grpo_loss(
    new_logprobs,
    old_logprobs,
    mask,
    advantage: float,
    *,
    clip: float = 0.2,
):
    import torch

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    surr = torch.minimum(ratio * advantage, clipped * advantage)
    active = mask.float()
    denom = active.sum().clamp(min=1.0)
    return -(surr * active).sum() / denom


def _forward_response_logprobs(model, prompt_ids: Sequence[int], response_ids: Sequence[int], device: str):
    import torch

    ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
    logits = model(input_ids=ids).logits
    start = max(len(prompt_ids) - 1, 0)
    step_logits = logits[0, start : start + len(response_ids)]
    logp = torch.nn.functional.log_softmax(step_logits, dim=-1)
    token_ids = torch.tensor(list(response_ids), device=device)
    gathered = logp.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return gathered


def apply_hf_grpo_update(policy: Any, groups: list[dict[str, Any]], cfg: GRPOConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW

    ensure = getattr(policy, "_ensure_loaded", None)
    if callable(ensure):
        ensure()
    model = getattr(policy, "_model", None)
    device = getattr(policy, "_device", None) or "cpu"
    if model is None:
        return {"status": "collected_no_update", "error": "HuggingFace policy has no loaded weights"}
    if cfg.adapter and Path(cfg.adapter).exists():
        from peft import PeftModel

        if not hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(model, cfg.adapter, is_trainable=True)
            policy._model = model
        elif not any(p.requires_grad for p in model.parameters()):
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            model = PeftModel.from_pretrained(base, cfg.adapter, is_trainable=True)
            policy._model = model
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return {"status": "collected_no_update", "error": "no trainable LoRA parameters attached"}
    model.train()
    optimizer = AdamW(trainable, lr=cfg.learning_rate)
    updates = 0
    last_loss = None
    for group in groups:
        advantages = group.get("advantages") or []
        for traj, advantage in zip(group.get("trajectories") or [], advantages):
            if updates >= cfg.max_updates:
                break
            if not traj.get("usable_for_rl"):
                continue
            prompt_ids = traj.get("prompt_ids") or []
            response_ids = traj.get("response_ids") or []
            old = traj.get("response_logprobs") or []
            mask_raw = traj.get("response_mask") or [1] * len(response_ids)
            if not prompt_ids or not response_ids or not old:
                continue
            limit = min(len(response_ids), len(old), len(mask_raw))
            new_lp = _forward_response_logprobs(model, prompt_ids, response_ids[:limit], device)
            old_lp = torch.tensor([float(v) for v in old[:limit]], device=device)
            mask = torch.tensor([int(v) for v in mask_raw[:limit]], device=device)
            loss = tensor_grpo_loss(new_lp, old_lp, mask, float(advantage), clip=cfg.clip)
            optimizer.zero_grad(set_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            updates += 1
        if updates >= cfg.max_updates:
            break
    out = Path(cfg.output_dir) / "adapter"
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out)
    return {
        "status": "ran" if updates else "collected_no_update",
        "updates": updates,
        "last_loss": last_loss,
        "adapter": str(out),
        "device": device,
    }


async def collect_group(
    task: TaskInput,
    spec: GradingSpec,
    model: PolicyModel,
    tools: ToolDispatcher,
    *,
    group_id: str,
    group_size: int,
    cost_lambda: float = 0.0,
) -> dict[str, Any]:
    records: list[EpisodeRecord] = []
    rewards: list[float] = []
    for idx in range(group_size):
        record = await run_episode(task, model, tools, episode_id=f"{group_id}-{idx}")
        score = score_episode(
            record.result,
            spec,
            cost_lambda=cost_lambda,
            max_explore=task.budget.max_explore_calls,
        )
        records.append(record)
        rewards.append(float(score.reward))
    group = export_grpo_group(records, rewards, group_id=group_id)
    group["advantages"] = group_advantages(rewards)
    group["task_id"] = task.public_id()
    group["policy_version"] = getattr(model, "policy_version", group["policy_version"])
    return group


def choose_backend(requested: str, compat: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    if compat.get("cuda_available") and compat.get("verl") not in {None, "not_installed"}:
        return "verl"
    if compat.get("cuda_available") or compat.get("mps_available") or (
        compat.get("torch") not in {None, "not_installed"}
    ):
        return "huggingface"
    return "none"


async def _collect_groups(
    tasks: list[TaskInput],
    specs: dict[str, GradingSpec],
    model: PolicyModel,
    tools: ToolDispatcher,
    cfg: GRPOConfig,
) -> list[dict[str, Any]]:
    groups = []
    for task in tasks[: cfg.max_prompts]:
        if task.split == "test":
            continue
        spec = specs[task.public_id()]
        group = await collect_group(
            task,
            spec,
            model,
            tools,
            group_id=f"grpo-{task.public_id()}",
            group_size=cfg.group_size,
            cost_lambda=cfg.cost_lambda,
        )
        groups.append(group)
    return groups


def run_grpo(
    cfg: GRPOConfig,
    *,
    tasks: list[TaskInput] | None = None,
    specs: dict[str, GradingSpec] | None = None,
    model: PolicyModel | None = None,
    tools: ToolDispatcher | None = None,
) -> dict[str, Any]:
    compat = write_report()
    backend = choose_backend(cfg.backend, compat)
    result: dict[str, Any] = {
        "status": "not_run",
        "backend": backend,
        "compat": compat,
        "group_size": cfg.group_size,
        "clip": cfg.clip,
    }
    if backend == "verl":
        result["status"] = "not_run"
        result["error"] = (
            "verl is the training framework. Start GRPO with scripts/train/verl_grpo.sh "
            "(python -m verl.trainer.main_ppo + VerlResearchAgentLoop). "
            "This module only collects groups or runs a HuggingFace LoRA fallback."
        )
        result["hint"] = "scripts/train/verl_grpo.sh"
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / "grpo_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    if tasks is None or specs is None or tools is None:
        from research_agent.evaluation.runner import snapshot_from_dir

        data_dir = Path(cfg.data)
        if not (data_dir / "public" / "tasks.jsonl").exists():
            result["error"] = f"missing prepared data at {data_dir}"
            return result
        tasks, specs, tools = snapshot_from_dir(data_dir)
    if model is None:
        if not cached_hf_model(cfg.model_name):
            result["error"] = (
                f"{cfg.model_name} weights are not on disk; Ollama cannot supply GRPO logprobs. "
                "Load HF weights on the GPU node."
            )
            result["hint"] = "Ollama generations lack token logprobs and cannot update GRPO"
            (Path(cfg.output_dir)).mkdir(parents=True, exist_ok=True)
            (Path(cfg.output_dir) / "grpo_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
        from research_agent.models.factory import load_policy

        model = load_policy(
            "huggingface",
            model_name=cfg.model_name,
            adapter=cfg.adapter if cfg.adapter and Path(cfg.adapter).exists() else None,
            local_files_only=True,
            trainable_adapter=True,
        )
    groups = asyncio.run(_collect_groups(tasks, specs, model, tools, cfg))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out / "groups.jsonl", groups)
    all_fail = sum(1 for group in groups if group.get("all_fail"))
    result.update(
        {
            "status": "collected",
            "n_groups": len(groups),
            "all_fail_groups": all_fail,
            "mean_reward": (
                sum(float(group["mean_reward"]) for group in groups) / len(groups) if groups else 0.0
            ),
            "zero_advantage_groups": sum(1 for group in groups if group.get("zero_advantage_variance")),
        }
    )
    if backend == "huggingface":
        usable = [
            row
            for group in groups
            for row in group["trajectories"]
            if row.get("usable_for_rl") and row.get("response_logprobs")
        ]
        if not usable:
            result["status"] = "collected_no_update"
            result["error"] = "no usable_for_rl logprobs; cannot run a GRPO update from this policy"
        else:
            result["usable_trajectories"] = len(usable)
            result["status"] = "collected_ready_for_update"
            result["note"] = (
                "A parameter update still requires the HuggingFace model + LoRA adapter in-process. "
                "This collection proves group construction, masks, and advantages."
            )
            try:
                updated = apply_hf_grpo_update(model, groups, cfg)
                result.update(updated)
            except Exception as exc:  # pragma: no cover - depends on GPU stack
                result["status"] = "collected_no_update"
                result["error"] = f"GRPO update failed: {exc}"
    (out / "grpo_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/grpo.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--backend", default=None)
    args = parser.parse_args(argv)
    cfg = load_grpo_config(Path(args.config))
    if args.data:
        cfg.data = args.data
    if args.output:
        cfg.output_dir = args.output
    if args.backend:
        cfg.backend = args.backend
    result = run_grpo(cfg)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"ran", "collected", "collected_ready_for_update"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
