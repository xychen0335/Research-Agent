#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIRD_TRAIN_JSON=${BIRD_TRAIN_JSON:-${ROOT}/data/splits/rl_train.json}
BIRD_VAL_JSON=${BIRD_VAL_JSON:-${ROOT}/data/splits/rl_val.json}
BIRD_SFT_TRAIN_JSON=${BIRD_SFT_TRAIN_JSON:-${ROOT}/data/splits/sft_train.json}
BIRD_SFT_VAL_JSON=${BIRD_SFT_VAL_JSON:-${ROOT}/data/splits/sft_val.json}
BIRD_DB_DIR=${BIRD_DB_DIR:-${ROOT}/data/databases/train}
BIRD_COLUMN_MEANING=${BIRD_COLUMN_MEANING:-${ROOT}/data/raw/bird23-train-filtered/column_meaning.json}
PYTHON=${PYTHON:-${ROOT}/upstream/verl/.venv/bin/python3}

mkdir -p "${ROOT}/data/processed"
[[ -x "${PYTHON}" ]] || { echo "Run scripts/setup_autodl.sh first." >&2; exit 1; }
cd "${ROOT}"
"${PYTHON}" -m agenticrl.bird_data --mode rl --data "${BIRD_TRAIN_JSON}" \
  --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
  --output "${ROOT}/data/processed/bird_rl_train.jsonl" "${@}"
"${PYTHON}" -m agenticrl.bird_data --mode rl --data "${BIRD_VAL_JSON}" \
  --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
  --output "${ROOT}/data/processed/bird_rl_val.jsonl" "${@}"
TOKEN_CHECK_INPUTS=(
  --input "${ROOT}/data/processed/bird_rl_train.jsonl"
  --input "${ROOT}/data/processed/bird_rl_val.jsonl"
)

if [[ -n "${BIRD_SFT_TRAJECTORIES:-}" ]]; then
  : "${BIRD_SFT_EVALUATION:?Set BIRD_SFT_EVALUATION to the matching eval_results.json}"
  "${PYTHON}" -m agenticrl.bird_data --mode sft --data "${BIRD_SFT_TRAIN_JSON}" \
    --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
    --trajectories "${BIRD_SFT_TRAJECTORIES}" \
    --evaluation "${BIRD_SFT_EVALUATION}" \
    --output "${ROOT}/data/processed/bird_sft_train.parquet" "${@}"
  TOKEN_CHECK_INPUTS+=(--input "${ROOT}/data/processed/bird_sft_train.parquet")
fi

if [[ -n "${BIRD_SFT_VAL_TRAJECTORIES:-}" ]]; then
  : "${BIRD_SFT_VAL_EVALUATION:?Set BIRD_SFT_VAL_EVALUATION to the matching eval_results.json}"
  "${PYTHON}" -m agenticrl.bird_data --mode sft --data "${BIRD_SFT_VAL_JSON}" \
    --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
    --trajectories "${BIRD_SFT_VAL_TRAJECTORIES}" \
    --evaluation "${BIRD_SFT_VAL_EVALUATION}" \
    --output "${ROOT}/data/processed/bird_sft_val.parquet" "${@}"
  TOKEN_CHECK_INPUTS+=(--input "${ROOT}/data/processed/bird_sft_val.parquet")
fi

"${PYTHON}" "${ROOT}/scripts/prepare_eval_tasks.py"
"${PYTHON}" "${ROOT}/scripts/check_token_lengths.py" \
  "${TOKEN_CHECK_INPUTS[@]}" \
  --model "${ROOT}/models/Qwen3.5-4B-851bf6e8" \
  --max-length 8192 \
  --output "${ROOT}/data/processed/token_lengths.json"
