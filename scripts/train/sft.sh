#!/usr/bin/env bash
# Optional LoRA SFT on a messages jsonl. Not part of the default GRPO path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m research_agent.training.compat || true
python -m research_agent.training.sft --config configs/training/sft.yaml "$@"
