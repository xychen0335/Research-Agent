#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${TEACHER_MODEL_PATH:?Set TEACHER_MODEL_PATH to a local or Hugging Face model}"
: "${BIRD_DATA_JSON:?Set BIRD_DATA_JSON to the training split used for trajectory generation}"
: "${BIRD_DB_DIR:?Set BIRD_DB_DIR to the matching database directory}"
: "${BIRD_COLUMN_MEANING:?Set BIRD_COLUMN_MEANING to column_meaning.json}"
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/outputs/teacher-trajectories}

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
