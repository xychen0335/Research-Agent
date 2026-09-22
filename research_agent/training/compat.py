"""Probe GPU / verl / vLLM presence. Does not run a training step."""

from __future__ import annotations

import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from research_agent.paths import DEFAULT_MODEL, model_available, resolve_model_path


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def cached_hf_model(name: str | None = None) -> bool:
    return model_available(name)


def probe() -> dict[str, Any]:
    model_path = resolve_model_path()
    report: dict[str, Any] = {
        "torch": None,
        "cuda_available": False,
        "cuda_device": None,
        "mps_available": False,
        "vllm": None,
        "verl": None,
        "gpu_training": "not_run",
        "qwen35_4b": "unverified",
        "qwen35_4b_cached": model_available(),
        "model_path": str(model_path),
    }
    torch_version = _package_version("torch")
    if torch_version is None:
        report["torch"] = "not_installed"
    else:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
    if report["qwen35_4b_cached"]:
        report["qwen35_4b"] = "local"
    report["vllm"] = _package_version("vllm") or "not_installed"
    report["verl"] = _package_version("verl") or "not_installed"
    usage = shutil.disk_usage(Path.cwd())
    report["disk_free_gb"] = round(usage.free / (1024**3), 2)
    return report


def write_report(path: Path | None = None) -> dict[str, Any]:
    report = probe()
    dest = path or Path("outputs/gpu_compat.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = write_report()
    print(json.dumps(report, indent=2))
    if report.get("cuda_available") and report.get("verl") not in {None, "not_installed"}:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
