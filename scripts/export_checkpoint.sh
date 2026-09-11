#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a verl global_step directory. For GRPO, append /actor.}"
: "${TARGET_DIR:?Set TARGET_DIR for the exported Hugging Face model or adapter}"
VERL_ROOT="${ROOT}/upstream/verl"

export PYTHONPATH="${ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
cd "${VERL_ROOT}"
uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.model_merger merge \
  --backend fsdp --local_dir "${CHECKPOINT_DIR}" --target_dir "${TARGET_DIR}"
