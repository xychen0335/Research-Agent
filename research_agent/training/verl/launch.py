"""Flatten Hydra overrides and start verl.trainer.main_ppo.

GRPO, FSDP, vLLM weight sync, and the PPO step belong to verl. This module only
builds the CLI that points the framework at our Agent Loop and reward function.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from research_agent.training.compat import write_report
from research_agent.training.export import write_verl_files

DEFAULT_CONFIG = Path("configs/training/verl_grpo.yaml")
SKIP_KEYS = {"status", "note", "framework_entry"}

_RESOLVE_KEYS = (
    "agent_loop_config_path",
    "custom_reward_function.path",
    "train_files",
    "val_files",
    "research_agent.data",
    "rollout.agent.data",
    "lora_adapter_path",
    "model.path",
)


def _flatten(payload: Any, prefix: str = "") -> list[str]:
    if not isinstance(payload, dict):
        if prefix:
            return [f"{prefix}={_format_value(payload)}"]
        return []
    rows: list[str] = []
    for key, value in payload.items():
        if key in SKIP_KEYS:
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten(value, path))
        else:
            rows.append(f"{path}={_format_value(value)}")
    return rows


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def hydra_overrides(raw: dict[str, Any], *, root: Path) -> list[str]:
    rows = _flatten(raw)
    rewritten: list[str] = []
    for item in rows:
        key, sep, value = item.partition("=")
        if not sep:
            rewritten.append(item)
            continue
        if any(key.endswith(suffix) for suffix in _RESOLVE_KEYS) and value and not Path(value).is_absolute():
            value = str((root / value).resolve())
        if key == "research_agent.data":
            key = "+research_agent.data"
        rewritten.append(f"{key}={value}")
    return rewritten


def load_overlay(path: Path | None = None) -> dict[str, Any]:
    dest = path or DEFAULT_CONFIG
    loaded = yaml.safe_load(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    return loaded if isinstance(loaded, dict) else {}


def verl_main_ppo_argv(
    *,
    root: Path | None = None,
    overlay: Path | None = None,
    extra: list[str] | None = None,
    adapter: str | None = None,
) -> list[str]:
    root = (root or Path.cwd()).resolve()
    raw = load_overlay(overlay)
    extras = list(extra or [])
    if adapter:
        extras.append(f"+actor_rollout_ref.model.lora_adapter_path={Path(adapter).resolve()}")
    return hydra_overrides(raw, root=root) + extras


def _compat_blocks_launch(compat: dict[str, Any]) -> str | None:
    if compat.get("verl") in {None, "not_installed"}:
        return "verl is not installed; install the GPU stack on the training node"
    if not compat.get("cuda_available"):
        return "no CUDA device; refuse to start verl.trainer.main_ppo on this machine"
    return None


def run_verl_grpo(
    *,
    output_dir: Path | None = None,
    adapter: str | None = None,
    extra: list[str] | None = None,
    overlay: Path | None = None,
    data_dir: Path | None = None,
    exec_process: bool = False,
) -> dict[str, Any]:
    root = Path.cwd().resolve()
    out = Path(output_dir or os.environ.get("OUTPUT", "outputs/verl-grpo"))
    out.mkdir(parents=True, exist_ok=True)
    compat = write_report(out / "gpu_compat.json")
    report: dict[str, Any] = {
        "status": "not_run",
        "backend": "verl",
        "framework_entry": "verl.trainer.main_ppo",
        "compat": compat,
        "note": "This process does not implement GRPO; it only starts verl.",
    }
    blocked = _compat_blocks_launch(compat)
    if blocked:
        report["error"] = blocked
        report["hint"] = "scripts/train/verl_grpo.sh"
        (out / "verl_grpo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    prepared = Path(data_dir or os.environ.get("RESEARCH_AGENT_DATA", "data/processed/papersearchqa"))
    exported = write_verl_files(prepared)
    report["dataset"] = exported
    if exported.get("status") != "wrote":
        report["error"] = exported.get("error") or f"failed to export verl rows from {prepared}"
        (out / "verl_grpo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    adapter_path = adapter or os.environ.get("RESEARCH_AGENT_ADAPTER")
    overrides = verl_main_ppo_argv(root=root, overlay=overlay, extra=extra, adapter=adapter_path)
    report["overrides"] = overrides
    report["cmd"] = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    (out / "verl_grpo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["RESEARCH_AGENT_DATA"] = str(prepared.resolve())
    if adapter_path:
        env["RESEARCH_AGENT_ADAPTER"] = str(Path(adapter_path).resolve())
    if exec_process:
        os.execvpe(sys.executable, report["cmd"], env)
        return report
    completed = subprocess.run(report["cmd"], env=env, check=False)
    report["exit_code"] = int(completed.returncode)
    report["status"] = "ran" if completed.returncode == 0 else "failed"
    if completed.returncode != 0:
        report["error"] = f"verl.trainer.main_ppo exited {completed.returncode}"
    (out / "verl_grpo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    report = run_verl_grpo(extra=extra, exec_process=True)
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "ran" else 2


if __name__ == "__main__":
    raise SystemExit(main())
