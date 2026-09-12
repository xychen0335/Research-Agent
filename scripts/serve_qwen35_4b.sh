#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_BIN=${VLLM_BIN:-}
if [[ -z "${VLLM_BIN}" ]] && command -v vllm >/dev/null; then
  VLLM_BIN=$(command -v vllm)
elif [[ -z "${VLLM_BIN}" ]] && [[ -x "${ROOT}/upstream/verl/.venv/bin/vllm" ]]; then
  VLLM_BIN="${ROOT}/upstream/verl/.venv/bin/vllm"
fi

if ! command -v nvidia-smi >/dev/null || [[ -z "${VLLM_BIN}" ]]; then
  echo "Requires a Linux NVIDIA GPU host with vLLM installed." >&2
  exit 1
fi

mkdir -p "${ROOT}/outputs"
nvidia-smi > "${ROOT}/outputs/gpu.txt"
"$(dirname "${VLLM_BIN}")/python" -m pip freeze > "${ROOT}/outputs/inference-packages.txt"

MODEL_PATH=${MODEL_PATH:-${ROOT}/models/Qwen3.5-4B-851bf6e8}
SERVED_NAME=${SERVED_NAME:-base}
PORT=${PORT:-8000}

LORA_ARGS=()
if [[ -n "${LORA_PATH:-}" ]]; then
  LORA_ARGS=(--enable-lora --lora-modules "${SERVED_NAME}=${LORA_PATH}")
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.8 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${LORA_ARGS[@]}"
