#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_ROOT="${ROOT}/upstream/verl"
MODEL_PATH=${MODEL_PATH:-${ROOT}/models/Qwen3.5-4B-851bf6e8}
TRAIN_FILE=${TRAIN_FILE:-${ROOT}/data/processed/bird_sft_train.parquet}
VAL_FILE=${VAL_FILE:-${ROOT}/data/processed/bird_sft_val.parquet}
SAVE_DIR=${SAVE_DIR:-${ROOT}/checkpoints/qwen35-4b-bird-sft}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-8192}

[[ -f "${TRAIN_FILE}" ]] || { echo "Missing TRAIN_FILE=${TRAIN_FILE}" >&2; exit 1; }
[[ -f "${VAL_FILE}" ]] || { echo "Missing VAL_FILE=${VAL_FILE}" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "Requires an NVIDIA GPU." >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "Missing pinned model at MODEL_PATH=${MODEL_PATH}" >&2; exit 1; }

export PYTHONPATH="${ROOT}:${ROOT}/upstream/BIRD-RL:${VERL_ROOT}:${PYTHONPATH:-}"
cd "${VERL_ROOT}"
uv run --frozen --all-packages --extra vllm --extra fsdp \
  torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=4 \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=16384 \
  data.max_length="${MAX_LENGTH}" \
  data.use_dynamic_bsz=False \
  data.messages_key=messages \
  data.tools_key=tools \
  data.ignore_input_ids_mismatch=True \
  data.truncation=error \
  data.num_workers=2 \
  optim.lr="${LR}" \
  engine=fsdp \
  model.path="${MODEL_PATH}" \
  model.use_remove_padding=False \
  model.enable_gradient_checkpointing=True \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.target_modules=all-linear \
  model.exclude_modules='.*visual.*' \
  checkpoint.save_contents='[model,optimizer,extra]' \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.project_name=agenticrl-bird \
  trainer.experiment_name=qwen35-4b-sft \
  trainer.logger='[console,tensorboard]' \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=after_each_epoch \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.seed=20260911 \
  trainer.resume_mode=auto \
  "${@}"
