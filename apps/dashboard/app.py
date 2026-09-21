"""Trajectory board. Unrun checkpoints stay unrun; scores are never filled with zeros."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research_agent.harness.trajectory import load_jsonl

STATIC_DIR = Path(__file__).resolve().parent / "static"


class LiveRequest(BaseModel):
    question: str
    run_id: str = "live"
    policy: str = "scripted"


PREFERRED_LEFT = ("psqa-qwen35-4b-base", "cpu-oracle", "cpu-dev")
PREFERRED_RIGHT = ("psqa-no-retrieval", "synth-no-retrieval", "psqa-fixed-rag", "cpu-no-retrieval")
SHOWCASE_TASK_ID = "cs-005"
SHOWCASE_LEFT = "cpu-oracle"
SHOWCASE_RIGHT = "synth-no-retrieval"
PLANNED_UNRUN = (
    {
        "run_id": "sft-qwen35-4b",
        "note": "LoRA SFT 未运行：31 条短答案教师轨迹已过门槛，但本机无 CUDA/HF 权重。",
    },
    {
        "run_id": "grpo-qwen35-4b",
        "note": "GRPO 未运行：没有 SFT adapter，Ollama 生成不含 token logprob。",
    },
)


def _run_dirs(outputs: Path) -> list[Path]:
    if not outputs.exists():
        return []
    dirs = [path for path in outputs.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda path: path.name)


def _read_run(path: Path) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    episodes = load_jsonl(path / "episodes.jsonl")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        unrun = False
    else:
        metrics = None
        unrun = not episodes
    return {
        "run_id": path.name,
        "path": str(path),
        "unrun": unrun,
        "metrics": metrics,
        "n_episodes": len(episodes),
        "policy_version": (metrics or {}).get("policy_version")
        or (episodes[0].get("result", {}).get("policy_version") if episodes else None),
        "live": (metrics or {}).get("live") if metrics else (episodes[0].get("live") if episodes else None),
        "failures": (metrics or {}).get("failures"),
        "answer_em": (metrics or {}).get("answer_em"),
        "note": (metrics or {}).get("note"),
    }


def _planned_run(item: dict[str, str]) -> dict[str, Any]:
    return {
        "run_id": item["run_id"],
        "path": None,
        "unrun": True,
        "metrics": None,
        "n_episodes": 0,
        "policy_version": None,
        "live": None,
        "failures": None,
        "answer_em": None,
        "note": item["note"],
    }


def _read_missing(run_id: str) -> dict[str, Any]:
    planned = next((item for item in PLANNED_UNRUN if item["run_id"] == run_id), None)
    if planned:
        summary = _planned_run(planned)
        summary["episodes"] = []
        return summary
    return {
        "run_id": run_id,
        "unrun": True,
        "metrics": None,
        "episodes": [],
        "note": "run directory missing",
    }


def _episode_by_task(run: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for item in run.get("episodes") or []:
        if item.get("task_id") == task_id:
            return item
    return None


def create_app(outputs_dir: Path | None = None) -> FastAPI:
    outputs = outputs_dir or Path("outputs")
    app = FastAPI(title="research-agent-dashboard")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        runs = [_read_run(path) for path in _run_dirs(outputs)]
        names = {item["run_id"] for item in runs}
        for planned in PLANNED_UNRUN:
            if planned["run_id"] not in names:
                runs.append(_planned_run(planned))
        names = {item["run_id"] for item in runs}
        left = next((name for name in PREFERRED_LEFT if name in names), runs[0]["run_id"] if runs else None)
        right = next((name for name in PREFERRED_RIGHT if name in names and name != left), None)
        if right is None and len(runs) > 1:
            right = next((item["run_id"] for item in runs if item["run_id"] != left), runs[1]["run_id"])
        return {"runs": runs, "preferred_left": left, "preferred_right": right}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        path = outputs / run_id
        if not path.exists():
            planned = next((item for item in PLANNED_UNRUN if item["run_id"] == run_id), None)
            if planned is None:
                raise HTTPException(404, "run not found")
            summary = _planned_run(planned)
            summary["episodes"] = []
            return summary
        summary = _read_run(path)
        episodes = load_jsonl(path / "episodes.jsonl")
        cards = []
        for row in episodes:
            result = row.get("result") or {}
            events = row.get("events") or []
            searches = [
                event.get("payload", {}).get("arguments", {}).get("query")
                for event in events
                if event.get("event_type") == "action" and event.get("payload", {}).get("name") == "search"
            ]
            cards.append(
                {
                    "episode_id": row.get("episode_id"),
                    "task_id": (row.get("task") or {}).get("task_id"),
                    "question": (row.get("task") or {}).get("question"),
                    "answer": result.get("answer"),
                    "citations": result.get("citations"),
                    "status": result.get("status"),
                    "termination": result.get("termination"),
                    "policy_version": result.get("policy_version"),
                    "usage": result.get("usage"),
                    "live": row.get("live", True),
                    "searches": searches,
                }
            )
        summary["episodes"] = cards
        return summary

    @app.get("/api/runs/{run_id}/episodes/{episode_id}")
    def get_episode(run_id: str, episode_id: str) -> dict[str, Any]:
        path = outputs / run_id / "episodes.jsonl"
        for row in load_jsonl(path):
            if row.get("episode_id") == episode_id:
                return row
        raise HTTPException(404, "episode not found")

    @app.get("/api/compare")
    def compare(left: str, right: str) -> dict[str, Any]:
        left_run = get_run(left)
        right_run = get_run(right)
        left_map = {item.get("task_id") or item.get("question"): item for item in left_run.get("episodes", [])}
        right_map = {item.get("task_id") or item.get("question"): item for item in right_run.get("episodes", [])}
        keys = sorted(set(left_map) | set(right_map))
        rows = []
        for key in keys:
            litem = left_map.get(key)
            ritem = right_map.get(key)
            rows.append(
                {
                    "task_id": key,
                    "question": (litem or ritem or {}).get("question"),
                    "left": litem if litem else {"unrun": True},
                    "right": ritem if ritem else {"unrun": True},
                }
            )
        return {"left": left, "right": right, "rows": rows, "left_metrics": left_run.get("metrics"), "right_metrics": right_run.get("metrics")}

    @app.get("/api/showcase")
    def showcase() -> dict[str, Any]:
        from research_agent.evaluation.planning import compare_planning_conditions

        left_run = get_run(SHOWCASE_LEFT) if (outputs / SHOWCASE_LEFT).exists() else _read_missing(SHOWCASE_LEFT)
        right_run = get_run(SHOWCASE_RIGHT) if (outputs / SHOWCASE_RIGHT).exists() else _read_missing(SHOWCASE_RIGHT)
        left_ep = _episode_by_task(left_run, SHOWCASE_TASK_ID)
        right_ep = _episode_by_task(right_run, SHOWCASE_TASK_ID)
        question = (left_ep or right_ep or {}).get("question")
        planning = compare_planning_conditions(task_id=SHOWCASE_TASK_ID, outputs=outputs)
        if planning.get("question"):
            question = planning["question"]
        return {
            "task_id": SHOWCASE_TASK_ID,
            "question": question,
            "left_run": SHOWCASE_LEFT,
            "right_run": SHOWCASE_RIGHT,
            "left": left_ep or {"unrun": True},
            "right": right_ep or {"unrun": True},
            "planning": planning,
            "note": (
                "Computer-science showcase on synthetic-dev. "
                "Scripted oracle vs no-retrieval; not a trained Qwen3.5-4B checkpoint."
            ),
        }

    @app.post("/api/live")
    async def live(payload: LiveRequest) -> dict[str, Any]:
        from research_agent.contracts import TOOL_SEARCH, TOOL_SUBMIT, Budget, TaskInput
        from research_agent.data.prepare import prepare_synthetic_dev
        from research_agent.environment.corpus import CorpusSnapshot
        from research_agent.environment.tools import ToolEnvironment
        from research_agent.harness.loop import run_episode
        from research_agent.models.scripted import ScriptedPolicy, tool_call

        prepared = Path("data/processed/synthetic-dev")
        if not (prepared / "public" / "corpus.jsonl").exists():
            prepare_synthetic_dev(prepared)
        policy = payload.policy or "scripted"
        if policy in {"ollama", "openai_compatible", "huggingface", "hf"}:
            from research_agent.models.factory import load_policy

            data_dir = Path("data/processed/papersearchqa-dev")
            if not (data_dir / "public" / "corpus.jsonl").exists():
                data_dir = prepared
                if not (prepared / "public" / "corpus.jsonl").exists():
                    prepare_synthetic_dev(prepared)
            corpus = CorpusSnapshot.from_jsonl(data_dir / "public" / "corpus.jsonl")
            tools = ToolEnvironment(corpus)
            model = load_policy(policy)
            env_id = "papersearchqa-dev" if data_dir.name == "papersearchqa-dev" else "synthetic-dev"
        else:
            corpus = CorpusSnapshot.from_jsonl(prepared / "public" / "corpus.jsonl")
            tools = ToolEnvironment(corpus)
            model = ScriptedPolicy(
                {
                    "*": [
                        tool_call(TOOL_SEARCH, {"query": payload.question}),
                        tool_call(TOOL_SUBMIT, {"answer": "unknown", "citations": []}),
                    ]
                },
                policy_version="scripted-live",
            )
            env_id = "synthetic-dev"
        task = TaskInput(
            request_id=payload.run_id,
            question=payload.question,
            environment_id=env_id,
            budget=Budget(),
        )
        record = await run_episode(task, model, tools, live=True)
        run_dir = outputs / payload.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = load_jsonl(run_dir / "episodes.jsonl")
        existing.append(record.to_dict())
        with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in existing:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return record.public_summary()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
