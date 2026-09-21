"""Compare frozen evaluation runs. Missing checkpoints stay unrun; scores are not zero-filled."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def summarize_run(path: Path) -> dict[str, Any]:
    metrics = _load_json(path / "metrics.json")
    resolved = _load_json(path / "resolved_config.json")
    if metrics is None:
        return {
            "run_id": path.name,
            "path": str(path),
            "status": "not_run",
            "answer_em": None,
            "n": None,
            "policy_version": None,
            "harness_version": (resolved or {}).get("harness_version") if resolved else None,
            "note": "metrics.json missing; not filled with zeros",
        }
    return {
        "run_id": path.name,
        "path": str(path),
        "status": "ran",
        "answer_em": metrics.get("answer_em"),
        "legal_citation_rate": metrics.get("legal_citation_rate"),
        "submit_rate": metrics.get("submit_rate"),
        "n": metrics.get("n") or metrics.get("n_tasks"),
        "policy_version": metrics.get("policy_version"),
        "harness_version": metrics.get("harness_version") or (resolved or {}).get("harness_version"),
        "environment_version": metrics.get("environment_version") or (resolved or {}).get("environment_version"),
        "live": metrics.get("live"),
    }


def compare_frozen_runs(run_dirs: list[Path]) -> dict[str, Any]:
    runs = [summarize_run(Path(path)) for path in run_dirs]
    harness = {item.get("harness_version") for item in runs if item.get("status") == "ran"}
    harness.discard(None)
    return {
        "runs": runs,
        "same_harness": len(harness) <= 1,
        "harness_versions": sorted(str(item) for item in harness),
        "note": "Unrun checkpoints stay unrun. Do not treat missing metrics as 0.0.",
    }


def write_compare_report(run_dirs: list[Path], output: Path) -> dict[str, Any]:
    report = compare_frozen_runs(run_dirs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
