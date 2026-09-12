"""Fail fast unless the host matches the pinned A100 training environment."""

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def os_release():
    values = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    return values


def check(strict=False):
    config = json.loads((ROOT / "configs/environment.json").read_text(encoding="utf-8"))
    errors = []
    facts = {"platform": sys.platform, "machine": platform.machine(), "python": platform.python_version()}
    if sys.platform != config["platform"]:
        errors.append("platform must be linux")
    if platform.machine() != config["machine"]:
        errors.append("machine must be x86_64")
    if not platform.python_version().startswith(config["python"] + "."):
        errors.append("Python must be 3.12.x")
    release = os_release()
    facts["os_release"] = release.get("PRETTY_NAME")
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        errors.append("OS must be Ubuntu 24.04")
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True).strip().split()[1]
    except Exception as exc:
        uv_version = None
        errors.append("uv check failed: " + str(exc))
    facts["uv"] = uv_version
    if uv_version != config["uv"]:
        errors.append("uv must be version " + config["uv"])
    verl_root = ROOT / "upstream/verl"
    try:
        verl_commit = subprocess.check_output(
            ["git", "-C", str(verl_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        verl_commit = None
        errors.append("verl revision check failed: " + str(exc))
    facts["verl_commit"] = verl_commit
    if verl_commit != config["upstream_verl_commit"]:
        errors.append("upstream/verl is not at the pinned commit")
    lock_path = ROOT / "upstream/verl/uv.lock"
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
    facts["uv_lock_sha256"] = lock_hash
    if lock_hash != config["uv_lock_sha256"]:
        errors.append("upstream/verl/uv.lock does not match the pinned checksum")

    try:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        name, memory, driver = [part.strip() for part in query.split(",")]
        facts.update({"gpu": name, "gpu_memory_mib": int(memory), "nvidia_driver": driver})
        if config["gpu_name_contains"] not in name:
            errors.append("GPU must be an A100")
        if int(memory) < config["minimum_gpu_memory_mib"]:
            errors.append("GPU memory is below 79,000 MiB")
        if int(driver.split(".")[0]) < config["minimum_nvidia_driver_major"]:
            errors.append("NVIDIA driver must be version 580 or newer for cu130")
    except Exception as exc:
        errors.append("nvidia-smi check failed: " + str(exc))

    installed = {}
    for module_name, expected in config["packages"].items():
        try:
            module = importlib.import_module(module_name)
            actual = getattr(module, "__version__", None)
            installed[module_name] = actual
            if strict and actual != expected:
                errors.append("%s=%s, expected %s" % (module_name, actual, expected))
        except Exception as exc:
            installed[module_name] = None
            if strict:
                errors.append("cannot import %s: %s" % (module_name, exc))
    facts["packages"] = installed
    try:
        import torch
        facts["torch_cuda"] = torch.version.cuda
        facts["cuda_available"] = torch.cuda.is_available()
        if strict and (torch.version.cuda != config["cuda_wheel_family"] or not torch.cuda.is_available()):
            errors.append("torch must expose a working CUDA 13.0 runtime")
    except Exception:
        pass
    return facts, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="also require every pinned Python package")
    args = parser.parse_args()
    facts, errors = check(args.strict)
    print(json.dumps({"facts": facts, "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
