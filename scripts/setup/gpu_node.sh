#!/usr/bin/env bash
# A100/Linux entry: install train extras if needed, refuse missing CUDA/weights, run gpu_mini.
# Does not download Qwen3.5-4B.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python scripts/setup/check_env.py
python -c "from research_agent.training.compat import probe; r=probe(); assert r.get('cuda_available'), 'gpu_node.sh requires CUDA'; assert r.get('qwen35_4b_cached'), 'Qwen/Qwen3.5-4B is not on disk; copy the HF snapshot before training'; print('cuda_ok')"
python -m pip install -e ".[data,train]"
# Local editable verl checkout (optional; train extra already pulls git+verl):
#   bash scripts/setup/install_verl.sh
python -m research_agent.training.pipeline \
  --experiment configs/experiments/gpu_mini.yaml \
  --output "${OUTPUT:-outputs/gpu-mini}"
# GRPO with the training framework (not research_agent.training.grpo):
#   scripts/train/verl_grpo.sh
