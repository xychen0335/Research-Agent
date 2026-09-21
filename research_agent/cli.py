"""CLI assembly. Business logic lives in research_agent.* modules."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from research_agent import HARNESS_VERSION, PACKAGE_VERSION
from research_agent.contracts import TaskInput


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _budget_from_cfg(raw: dict[str, Any]):
    from research_agent.contracts import Budget

    return Budget(
        max_explore_calls=int(raw.get("max_explore_calls", 6)),
        max_submit_calls=int(raw.get("max_submit_calls", 1)),
        max_observation_chars=int(raw.get("max_observation_chars", 4000)),
        max_context_tokens=int(raw.get("max_context_tokens", 8192)),
        max_generation_tokens=int(raw.get("max_generation_tokens", 1024)),
        search_topk=int(raw.get("search_topk", 3)),
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    from research_agent.data.prepare import prepare_source

    prepared = prepare_source(
        args.source,
        Path(args.output),
        input_path=Path(args.input) if args.input else None,
        split=args.split,
        n_train=args.n_train,
        n_test=args.n_test,
        n_distractors=args.n_distractors,
        seed=args.seed,
    )
    print(json.dumps(prepared.report, ensure_ascii=False, indent=2))
    return 0 if prepared.report.get("validation", {}).get("ok", False) else 1


def _select_tasks(
    tasks: list[TaskInput],
    *,
    split: str | None,
    limit: int | None,
    frozen_ids: list[str] | None = None,
) -> list[TaskInput]:
    selected = list(tasks)
    if split:
        selected = [task for task in selected if task.split == split]
    if frozen_ids:
        by_id = {task.public_id(): task for task in selected}
        selected = [by_id[task_id] for task_id in frozen_ids if task_id in by_id]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def cmd_eval(args: argparse.Namespace) -> int:
    from research_agent.data.prepare import prepare_synthetic_dev
    from research_agent.evaluation.baselines import run_baseline
    from research_agent.evaluation.runner import run_evaluation, snapshot_from_dir
    from research_agent.evaluation.snapshot import write_run_snapshot
    from research_agent.models.factory import load_policy

    prepared_dir = Path(args.data)
    if not (prepared_dir / "public" / "tasks.jsonl").exists():
        prepare_synthetic_dev(prepared_dir)
    tasks, specs, tools = snapshot_from_dir(prepared_dir)
    eval_cfg = _load_yaml(Path(args.eval_config)) if args.eval_config else {}
    frozen_ids = list(eval_cfg.get("frozen_task_ids") or [])
    split = args.split if args.split is not None else (eval_cfg.get("splits") or [None])[0]
    limit = args.limit if args.limit is not None else eval_cfg.get("limit")
    tasks = _select_tasks(tasks, split=split, limit=limit, frozen_ids=frozen_ids)
    output = Path(args.output)
    live_policies = {"openai_compatible", "openai", "ollama", "huggingface", "hf"}
    if args.policy in live_policies and args.baseline == "agent" and not args.oracle:
        model = load_policy(
            args.policy,
            model_name=args.model_name,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            adapter=getattr(args, "adapter", None),
            local_files_only=bool(getattr(args, "local_files_only", False)),
        )
        result = asyncio.run(
            run_evaluation(
                tasks,
                specs,
                model,
                tools,
                run_id=args.run_id,
                output_dir=output,
                live=True,
                max_concurrency=args.max_concurrency,
            )
        )
    else:
        result = asyncio.run(
            run_baseline(
                args.baseline,
                tasks,
                specs,
                tools,
                run_id=args.run_id,
                output_dir=output,
                oracle=args.oracle,
            )
        )

    cli_payload = {
        key: value
        for key, value in vars(args).items()
        if key != "func" and not callable(value)
    }
    write_run_snapshot(
        output,
        run_id=args.run_id,
        metrics=result.metrics,
        tasks=tasks,
        records=result.records,
        eval_cfg=eval_cfg,
        cli=cli_payload,
        extra={
            "baseline": args.baseline,
            "policy": args.policy,
            "data": args.data,
        },
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from research_agent.contracts import TOOL_SEARCH, TOOL_SUBMIT, TaskInput, to_plain
    from research_agent.data.prepare import prepare_synthetic_dev
    from research_agent.environment.corpus import CorpusSnapshot
    from research_agent.environment.tools import ToolEnvironment
    from research_agent.harness.loop import run_episode
    from research_agent.harness.trajectory import dump_jsonl
    from research_agent.models.scripted import ScriptedPolicy, tool_call

    prepared_dir = Path(args.data)
    if not (prepared_dir / "public" / "corpus.jsonl").exists():
        prepare_synthetic_dev(prepared_dir)
    corpus = CorpusSnapshot.from_jsonl(prepared_dir / "public" / "corpus.jsonl")
    tools = ToolEnvironment(corpus)
    task = TaskInput(
        request_id=args.request_id,
        question=args.question,
        environment_id="synthetic-dev",
        budget=_budget_from_cfg(_load_yaml(Path(args.harness)) if args.harness else {}),
    )
    if args.model == "scripted":
        model = ScriptedPolicy(
            {"*": [tool_call(TOOL_SEARCH, {"query": args.question}), tool_call(TOOL_SUBMIT, {"answer": "unknown", "citations": []})]},
            policy_version="scripted-cli",
        )
    else:
        from research_agent.models.factory import load_policy

        kind = "ollama" if args.model in {"qwen3.5:4b", "ollama"} else args.model
        model = load_policy(kind if kind in {"ollama", "openai_compatible", "huggingface"} else "openai_compatible", model_name=args.model)
    record = asyncio.run(run_episode(task, model, tools, live=True))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output / "episodes.jsonl", [record.to_dict()])
    print(json.dumps(to_plain(record.result), ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from research_agent.evaluation.compare import write_compare_report

    run_dirs = [Path(item.strip()) for item in str(args.runs).split(",") if item.strip()]
    report = write_compare_report(run_dirs, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ran = [item for item in report["runs"] if item.get("status") == "ran"]
    return 0 if ran else 2


def cmd_plan_review(args: argparse.Namespace) -> int:
    from research_agent.evaluation.planning import write_planning_review

    report = write_planning_review(Path(args.outputs), Path(args.output), task_id=args.task_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_eval_trained(args: argparse.Namespace) -> int:
    """Frozen eval of a trained adapter. Missing weights stay not_run."""
    adapter = Path(args.adapter) if args.adapter else None
    if adapter is None or not adapter.exists():
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "not_run",
            "error": "trained adapter missing; refusing to invent scores",
            "adapter": str(adapter) if adapter else None,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2
    args.baseline = "agent"
    args.policy = "huggingface"
    args.local_files_only = True
    return cmd_eval(args)


def cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    from apps.dashboard.app import create_app

    app = create_app(Path(args.outputs))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    prepare = sub.add_parser("prepare", help="Build public tasks, corpus, and private grading files")
    prepare.add_argument("--source", default="synthetic-dev")
    prepare.add_argument("--input")
    prepare.add_argument("--split", default="dev")
    prepare.add_argument("--output", default="data/processed/synthetic-dev")
    prepare.add_argument("--n-train", dest="n_train", type=int, default=40)
    prepare.add_argument("--n-test", dest="n_test", type=int, default=10)
    prepare.add_argument("--n-distractors", dest="n_distractors", type=int, default=40)
    prepare.add_argument("--seed", type=int, default=20260919)
    prepare.set_defaults(func=cmd_prepare)

    evaluate = sub.add_parser("eval", help="Run a baseline or live policy through the shared harness")
    evaluate.add_argument("--data", default="data/processed/synthetic-dev")
    evaluate.add_argument("--baseline", default="no_retrieval", choices=["no_retrieval", "fixed_rag", "agent"])
    evaluate.add_argument(
        "--policy",
        default="scripted",
        help="scripted, ollama, openai_compatible, or huggingface",
    )
    evaluate.add_argument("--model-name", dest="model_name", default="qwen3.5:4b")
    evaluate.add_argument("--base-url", dest="base_url", default=None)
    evaluate.add_argument("--split", default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--eval-config", dest="eval_config", default=None)
    evaluate.add_argument("--max-concurrency", dest="max_concurrency", type=int, default=1)
    evaluate.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    evaluate.add_argument("--adapter", default=None, help="Optional LoRA adapter directory for huggingface policy")
    evaluate.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    evaluate.add_argument("--oracle", action="store_true")
    evaluate.add_argument("--run-id", dest="run_id", default="cpu-dev")
    evaluate.add_argument("--output", default="outputs/cpu-dev")
    evaluate.set_defaults(func=cmd_eval)

    run = sub.add_parser("run", help="Run one question")
    run.add_argument("--question", required=True)
    run.add_argument("--data", default="data/processed/synthetic-dev")
    run.add_argument("--model", default="scripted")
    run.add_argument("--harness", default="configs/harness/default.yaml")
    run.add_argument("--request-id", dest="request_id", default="cli-request")
    run.add_argument("--output", default="outputs/cli-run")
    run.set_defaults(func=cmd_run)

    dash = sub.add_parser("dashboard", help="Serve the trajectory board")
    dash.add_argument("--outputs", default="outputs")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8765)
    dash.set_defaults(func=cmd_dashboard)

    compare = sub.add_parser("compare", help="Compare frozen eval directories without filling zeros")
    compare.add_argument("--runs", required=True, help="Comma-separated output directories")
    compare.add_argument("--output", default="outputs/compare-frozen.json")
    compare.set_defaults(func=cmd_compare)

    plan = sub.add_parser("plan-review", help="Frozen planner packets for cs-005; human scores stay empty")
    plan.add_argument("--outputs", default="outputs")
    plan.add_argument("--task-id", dest="task_id", default="cs-005")
    plan.add_argument("--output", default="outputs/planning-review.json")
    plan.set_defaults(func=cmd_plan_review)

    trained = sub.add_parser("eval-trained", help="Frozen eval of a LoRA adapter; missing weights stay not_run")
    trained.add_argument("--data", default="data/processed/papersearchqa-dev")
    trained.add_argument("--eval-config", dest="eval_config", default="configs/evaluation/base_mini.yaml")
    trained.add_argument("--model-name", dest="model_name", default="Qwen/Qwen3.5-4B")
    trained.add_argument("--adapter", required=True)
    trained.add_argument("--split", default=None)
    trained.add_argument("--limit", type=int, default=None)
    trained.add_argument("--max-concurrency", dest="max_concurrency", type=int, default=1)
    trained.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    trained.add_argument("--base-url", dest="base_url", default=None)
    trained.add_argument("--oracle", action="store_true")
    trained.add_argument("--run-id", dest="run_id", default="frozen-sft")
    trained.add_argument("--output", default="outputs/frozen-sft")
    trained.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    trained.set_defaults(func=cmd_eval_trained, baseline="agent", policy="huggingface")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version or not args.command:
        print(f"research-agent {PACKAGE_VERSION} harness {HARNESS_VERSION}")
        if not args.command:
            parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
