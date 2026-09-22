"""One measured GPU mini-run: GRPO from the base model, then freeze-eval instructions.

Does not download Qwen3.5-4B. Does not invent scores when the stack is missing.
SFT is optional and is not part of this path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_agent.training.compat import cached_hf_model, write_report
from research_agent.training.grpo import GRPOConfig, load_grpo_config, run_grpo
from research_agent.training.verl.launch import run_verl_grpo


def run_gpu_mini(
    *,
    grpo_config: Path | None = None,
    output_dir: Path | None = None,
    experiment: Path | None = None,
) -> dict[str, Any]:
    out = output_dir or Path("outputs/gpu-mini")
    out.mkdir(parents=True, exist_ok=True)
    exp: dict[str, Any] = {}
    if experiment and experiment.exists():
        import yaml

        loaded = yaml.safe_load(experiment.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            exp = loaded
    grpo_path = Path(exp["grpo"]) if exp.get("grpo") else (grpo_config or Path("configs/training/grpo.yaml"))
    verl_path = Path(exp["verl"]) if exp.get("verl") else Path("configs/training/verl_grpo.yaml")
    compat = write_report(out / "gpu_compat.json")
    grpo_cfg = load_grpo_config(grpo_path)
    base = grpo_cfg.model_name
    report: dict[str, Any] = {
        "status": "not_run",
        "compat": compat,
        "qwen35_4b_cached": cached_hf_model(base),
        "experiment": str(experiment) if experiment else None,
        "sft": None,
        "grpo": None,
        "verl": str(verl_path),
        "frozen_eval": {"status": "not_run"},
        "note": (
            "Plan vs CPU vs GPU must stay separate. This pipeline reports a GPU mini-run "
            "after a real GRPO step on Qwen3.5-4B from the base model."
        ),
    }
    if not (compat.get("cuda_available") or compat.get("mps_available")):
        report["error"] = "no CUDA/MPS device; gpu mini-run was not started"
        _write(out, report)
        return report
    if not cached_hf_model(base):
        report["error"] = f"{base} is missing. Place the HF snapshot at models/Qwen3.5-4B"
        _write(out, report)
        return report
    if compat.get("cuda_available") and compat.get("verl") not in {None, "not_installed"}:
        grpo = run_verl_grpo(
            output_dir=out / "grpo",
            adapter=None,
            overlay=verl_path,
            data_dir=Path(grpo_cfg.data),
            exec_process=False,
        )
    else:
        grpo_cfg.output_dir = str(out / "grpo")
        grpo_cfg.adapter = ""
        if compat.get("verl") not in {None, "not_installed"}:
            grpo_cfg.backend = "huggingface"
        grpo = run_grpo(grpo_cfg)
        if grpo.get("backend") == "verl":
            grpo["hint"] = "scripts/train/verl_grpo.sh"
    report["grpo"] = grpo
    adapter = grpo.get("adapter") if grpo.get("status") == "ran" and grpo.get("adapter") else None
    if adapter and Path(str(adapter)).exists():
        from argparse import Namespace

        from research_agent.cli import cmd_eval_trained

        eval_out = out / "frozen"
        code = cmd_eval_trained(
            Namespace(
                adapter=str(adapter),
                data="data/processed/papersearchqa-dev",
                eval_config="configs/evaluation/base_mini.yaml",
                model_name=base,
                split=None,
                limit=None,
                max_concurrency=1,
                max_tokens=1024,
                base_url=None,
                oracle=False,
                run_id="frozen-gpu-mini",
                output=str(eval_out),
                local_files_only=True,
                baseline="agent",
                policy="huggingface",
            )
        )
        metrics_path = eval_out / "metrics.json"
        report["frozen_eval"] = {
            "status": "ran" if code == 0 and metrics_path.exists() else "failed",
            "exit_code": code,
            "output": str(eval_out),
            "adapter": str(adapter),
        }
    else:
        report["frozen_eval"] = {
            "status": "not_run",
            "command": (
                "python -m research_agent.cli eval-trained "
                "--data data/processed/papersearchqa-dev "
                "--eval-config configs/evaluation/base_mini.yaml "
                f"--adapter {adapter} --run-id frozen-grpo --output outputs/frozen-grpo"
                if adapter
                else None
            ),
        }
    report["status"] = "ran" if grpo.get("status") == "ran" else str(grpo.get("status") or "not_run")
    _write(out, report)
    return report


def _write(out: Path, report: dict[str, Any]) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-config", default="configs/training/grpo.yaml")
    parser.add_argument("--experiment", default="configs/experiments/gpu_mini.yaml")
    parser.add_argument("--output", default="outputs/gpu-mini")
    args = parser.parse_args(argv)
    report = run_gpu_mini(
        grpo_config=Path(args.grpo_config),
        experiment=Path(args.experiment) if getattr(args, "experiment", None) else None,
        output_dir=Path(args.output),
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "ran" else 2


if __name__ == "__main__":
    raise SystemExit(main())
