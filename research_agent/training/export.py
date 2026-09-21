"""Convert harness trajectories into backend-neutral training rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_agent.contracts import TaskInput
from research_agent.grading.contracts import GradingSpec
from research_agent.harness.trajectory import EpisodeRecord


def export_sft_messages(record: EpisodeRecord) -> dict[str, Any]:
    return {
        "task_id": record.task.public_id(),
        "messages": record.messages,
        "policy_version": record.result.policy_version,
        "harness_version": record.result.harness_version,
        "environment_version": record.result.environment_version,
        "loss_on": "assistant",
    }


def export_grpo_group(records: list[EpisodeRecord], rewards: list[float], *, group_id: str) -> dict[str, Any]:
    if len(records) != len(rewards):
        raise ValueError("records and rewards length mismatch")
    versions = {record.result.policy_version for record in records}
    if len(versions) != 1:
        raise ValueError("GRPO group mixed policy versions")
    mean_r = sum(rewards) / len(rewards)
    rows = []
    for record, reward in zip(records, rewards):
        trace = record.token_trace
        rows.append(
            {
                "episode_id": record.episode_id,
                "group_id": group_id,
                "prompt_ids": trace.prompt_ids,
                "response_ids": trace.response_ids,
                "response_mask": trace.response_mask,
                "response_logprobs": trace.response_logprobs,
                "reward": reward,
                "advantage": reward - mean_r,
                "usable_for_rl": trace.usable_for_rl,
                "policy_version": record.result.policy_version,
                "harness_version": record.result.harness_version,
                "environment_version": record.result.environment_version,
                "termination": record.result.termination,
            }
        )
    all_fail = all(reward == 0 for reward in rewards)
    all_success = all(reward == 1 for reward in rewards)
    zero_var = len(set(rewards)) == 1
    return {
        "group_id": group_id,
        "policy_version": next(iter(versions)),
        "n": len(rows),
        "mean_reward": mean_r,
        "all_fail": all_fail,
        "all_success": all_success,
        "zero_advantage_variance": zero_var,
        "trajectories": rows,
    }


def export_verl_row(task: TaskInput, spec: GradingSpec) -> dict[str, Any]:
    """RLHFDataset-style row. Gold stays in reward_model / extra_info, never in prompt."""
    return {
        "data_source": "research-agent",
        "prompt": [{"role": "user", "content": task.question}],
        "ability": "research-qa",
        "agent_name": "research_agent",
        "reward_model": {"style": "rule", "ground_truth": spec.answer},
        "extra_info": {
            "split": task.split,
            "index": task.public_id(),
            "task_id": task.public_id(),
            "question": task.question,
            "environment_id": task.environment_id,
            "aliases": list(spec.aliases),
            "scoring": spec.scoring,
            "answerable": spec.answerable,
            "gold_evidence_ids": list(spec.gold_evidence_ids),
        },
    }


def write_verl_files(data_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    from research_agent.evaluation.runner import snapshot_from_dir

    prepared = Path(data_dir)
    if not (prepared / "public" / "tasks.jsonl").exists():
        return {"status": "missing_data", "data_dir": str(prepared)}
    tasks, specs, _tools = snapshot_from_dir(prepared)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for task in tasks:
        spec = specs.get(task.public_id())
        if spec is None:
            continue
        row = export_verl_row(task, spec)
        if task.split == "test":
            test_rows.append(row)
        else:
            train_rows.append(row)
    dest = Path(output_dir) if output_dir is not None else prepared / "public"
    dest.mkdir(parents=True, exist_ok=True)
    train_jsonl = dest / "verl_train.jsonl"
    test_jsonl = dest / "verl_test.jsonl"
    train_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows), encoding="utf-8")
    test_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in test_rows), encoding="utf-8")
    report: dict[str, Any] = {
        "status": "wrote",
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_jsonl": str(train_jsonl),
        "test_jsonl": str(test_jsonl),
    }
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for name, rows in (("verl_train.parquet", train_rows), ("verl_test.parquet", test_rows)):
            table = pa.Table.from_pylist(rows)
            path = dest / name
            pq.write_table(table, path)
            report[name] = str(path)
    except Exception as exc:
        report["parquet"] = f"skipped: {exc}"
    return report
