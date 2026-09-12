#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${ROOT}/models/Qwen3.5-4B-851bf6e8}
BIRD_DATA_JSON=${BIRD_DATA_JSON:-${ROOT}/data/splits/sft_train.json}
BIRD_DB_DIR=${BIRD_DB_DIR:-${ROOT}/data/databases/train}
BIRD_COLUMN_MEANING=${BIRD_COLUMN_MEANING:-${ROOT}/data/raw/bird23-train-filtered/column_meaning.json}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/outputs/teacher-trajectories}
[[ -d "${TEACHER_MODEL_PATH}" ]] || { echo "Missing TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH}" >&2; exit 1; }

export PYTHONPATH="${ROOT}/upstream/BIRD-RL:${PYTHONPATH:-}"
if [[ -d "${ROOT}/upstream/verl/.venv/bin" ]]; then
  export PATH="${ROOT}/upstream/verl/.venv/bin:${PATH}"
fi
cd "${ROOT}/upstream/BIRD-RL"
bash scripts/infer/run_bird_inference.sh \
  --model_path "${TEACHER_MODEL_PATH}" \
  --dev_data "${BIRD_DATA_JSON}" \
  --db_dir "${BIRD_DB_DIR}" \
  --column_meaning "${BIRD_COLUMN_MEANING}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpu "${GPU:-0}" \
  --max_turns "${MAX_TURNS:-6}" \
  --max_tokens "${MAX_TOKENS:-2048}" \
  --batch_size "${BATCH_SIZE:-16}" \
  "${@}"
