#!/usr/bin/env python
"""Teacher sampling entry. The teacher never reads gold labels."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from research_agent.data.prepare import prepare_synthetic_dev
from research_agent.data.teacher import generate_teacher_traces, refilter_teacher_rows, sft_row, tasks_without_kept
from research_agent.evaluation.baselines import _oracle_scripts
from research_agent.evaluation.runner import snapshot_from_dir
from research_agent.harness.trajectory import dump_jsonl, load_jsonl
from research_agent.models.factory import load_policy
from research_agent.models.scripted import ScriptedPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/papersearchqa-dev")
    parser.add_argument("--output", default="outputs/teacher")
    parser.add_argument("--policy", default="ollama")
    parser.add_argument("--model-name", dest="model_name", default="qwen3.5:4b")
    parser.add_argument("--base-url", dest="base_url", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--samples", type=int, default=1, help="Rejection samples per train task")
    parser.add_argument("--stop-when-kept", dest="stop_when_kept", action="store_true")
    parser.add_argument("--skip-kept", dest="skip_kept", action="store_true")
    parser.add_argument("--target-kept", dest="target_kept", type=int, default=None)
    parser.add_argument("--refilter", action="store_true", help="Rebuild short SFT rows from all.jsonl without calling the model")
    parser.add_argument("--allow-oracle", action="store_true")
    args = parser.parse_args()
    prepared = Path(args.data)
    if not (prepared / "public" / "tasks.jsonl").exists():
        prepare_synthetic_dev(prepared)
    tasks, specs, tools = snapshot_from_dir(prepared)
    tasks = [task for task in tasks if task.split != "test"]
    if args.refilter:
        out = Path(args.output)
        all_rows = load_jsonl(out / "all.jsonl")
        kept = refilter_teacher_rows(all_rows, specs)
        dump_jsonl(out / "sft_short.jsonl", kept)
        summary = {
            "n": len(all_rows),
            "kept": len(kept),
            "tasks": len({item.get("task_id") for item in kept}),
            "source": str(out / "all.jsonl"),
            "output": str(out / "sft_short.jsonl"),
            "note": "Alias-span harvest for SFT only. Eval exact-match is unchanged.",
        }
        (out / "sft_short_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit:
        tasks = tasks[: args.limit]
    if args.policy in {"scripted-oracle", "oracle"}:
        if not args.allow_oracle:
            raise SystemExit("oracle teacher reads gold; pass --allow-oracle only for harness smoke tests")
        model = ScriptedPolicy(_oracle_scripts(tasks, specs), policy_version="scripted-oracle-teacher")
    else:
        model = load_policy(args.policy, model_name=args.model_name, base_url=args.base_url)
    out = Path(args.output)
    kept = load_jsonl(out / "sft.jsonl") if args.append else []
    all_rows = load_jsonl(out / "all.jsonl") if args.append else []
    if args.skip_kept:
        tasks = tasks_without_kept(tasks, kept)
    remaining = None
    if args.target_kept is not None:
        remaining = max(0, args.target_kept - len(kept))
        if remaining == 0:
            print(json.dumps({"n": len(all_rows), "kept": len(kept), "note": "target already met"}, indent=2))
            return 0

    def _flush(sample) -> None:
        row = {
            **sft_row(sample),
            "kept": sample.kept,
            "reason": sample.reason,
            "answer_score": sample.answer_score,
        }
        all_rows.append(row)
        if sample.kept:
            kept.append(sft_row(sample))
        dump_jsonl(out / "sft.jsonl", kept)
        dump_jsonl(out / "all.jsonl", all_rows)
        reasons: dict[str, int] = {}
        for item in all_rows:
            reason = str(item.get("reason") or item.get("teacher_filter_reason") or "")
            reasons[reason] = reasons.get(reason, 0) + 1
        summary = {
            "n": len(all_rows),
            "kept": len(kept),
            "tasks": len({item.get("task_id") for item in all_rows}),
            "samples_per_task": args.samples,
            "stop_when_kept": args.stop_when_kept,
            "policy": getattr(model, "policy_version", args.policy),
            "reasons": reasons,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"n": summary["n"], "kept": summary["kept"], "last": sample.reason}, ensure_ascii=False), flush=True)

    asyncio.run(
        generate_teacher_traces(
            tasks,
            specs,
            model,
            tools,
            samples_per_task=args.samples,
            stop_when_kept=args.stop_when_kept,
            on_sample=_flush,
            max_kept=remaining,
        )
    )
    print((out / "summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
