#!/usr/bin/env bash
# HuggingFace GRPO fallback: same-question group collection and optional LoRA update.
# The verl trainer is scripts/train/verl_grpo.sh (python -m verl.trainer.main_ppo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m research_agent.training.compat || true
python -m research_agent.training.grpo --config configs/training/grpo.yaml "$@"
