#!/usr/bin/env bash
# GPU node mini-run: LoRA SFT, one GRPO update, then print the frozen-eval command.
# Refuses to download Qwen3.5-4B. Exit 2 means not_run, not a fake zero score.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m research_agent.training.compat || true
python -m research_agent.training.pipeline \
  --experiment configs/experiments/gpu_mini.yaml \
  --output "${OUTPUT:-outputs/gpu-mini}" \
  "$@"
