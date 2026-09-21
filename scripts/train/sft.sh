#!/usr/bin/env bash
# LoRA SFT entry. Starts only after teacher traces exist and a GPU/MPS device is present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m research_agent.training.compat || true
python -m research_agent.training.sft --config configs/training/sft.yaml "$@"
