#!/usr/bin/env python
"""Create a conda or venv runtime and print device facts."""

from __future__ import annotations

from pathlib import Path
import platform
import shutil
import sys


def ensure_project_on_path() -> None:
    """macOS may mark venv .pth files UF_HIDDEN; Python then skips them."""
    import os
    import stat

    root = Path(__file__).resolve().parents[2]
    marker = Path(sys.prefix) / "lib"
    for site_packages in marker.glob("python*/site-packages"):
        pth = site_packages / "aa_research_agent.pth"
        pth.write_text(str(root) + "\n", encoding="utf-8")
        for path in site_packages.glob("*.pth"):
            flags = getattr(path.stat(), "st_flags", 0)
            hidden = getattr(stat, "UF_HIDDEN", 0)
            if hidden and flags & hidden:
                os.chflags(path, flags & ~hidden)


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"conda={shutil.which('conda') or 'missing'}")
    print(f"uv={shutil.which('uv') or 'missing'}")
    ensure_project_on_path()
    from research_agent.training.compat import probe

    report = probe()
    print(f"torch={report['torch']}")
    print(f"cuda_available={report['cuda_available']}")
    print(f"mps_available={report.get('mps_available')}")
    print(f"qwen35_4b_cached={report.get('qwen35_4b_cached')}")
    print(f"cuda_device={report['cuda_device']}")
    print(f"vllm={report['vllm']}")
    print(f"verl={report['verl']}")
    print(f"gpu_training={report['gpu_training']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
