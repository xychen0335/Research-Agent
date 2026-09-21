"""Write resolved run configs. Secrets stay out of the snapshot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from research_agent import HARNESS_VERSION, PACKAGE_VERSION
from research_agent.contracts import TaskInput
from research_agent.evaluation.planning import export_planning_packets
from research_agent.harness.trajectory import EpisodeRecord, dump_jsonl


_SECRET_KEYS = ("key", "token", "secret", "password", "authorization")


def git_manifest(repo: Path | None = None) -> dict[str, Any]:
    root = repo or Path.cwd()
    payload: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        payload["commit"] = commit
        payload["dirty"] = bool(dirty.strip())
        payload["dirty_summary"] = dirty.strip().splitlines()[:40]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        payload["error"] = "git_unavailable"
    return payload


def redact_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        lowered = str(key).lower()
        if any(part in lowered for part in _SECRET_KEYS):
            cleaned[key] = "<redacted>"
        elif isinstance(value, dict):
            cleaned[key] = redact_mapping(value)
        else:
            cleaned[key] = value
    return cleaned


def write_run_snapshot(
    output_dir: Path,
    *,
    run_id: str,
    metrics: dict[str, Any],
    tasks: list[TaskInput],
    records: list[EpisodeRecord] | None = None,
    eval_cfg: dict[str, Any] | None = None,
    cli: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_ids = list((eval_cfg or {}).get("frozen_task_ids") or [])
    selected_ids = [task.public_id() for task in tasks]
    resolved = {
        "run_id": run_id,
        "package_version": PACKAGE_VERSION,
        "harness_version": HARNESS_VERSION,
        "n_tasks": len(tasks),
        "selected_task_ids": selected_ids,
        "frozen_task_ids": frozen_ids,
        "eval_config": redact_mapping(eval_cfg or {}),
        "cli": redact_mapping(cli or {}),
        "git": git_manifest(),
        "metrics_keys": sorted(metrics.keys()),
        **(extra or {}),
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = dict(metrics)
    if frozen_ids:
        metrics["frozen_task_ids"] = frozen_ids
    metrics["n_selected"] = len(tasks)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if records is not None:
        dump_jsonl(output_dir / "planning.jsonl", export_planning_packets(records))
    return resolved
