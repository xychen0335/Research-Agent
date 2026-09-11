#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${BIRD_TRAIN_JSON:?Set BIRD_TRAIN_JSON to the BIRD train JSON file}"
: "${BIRD_VAL_JSON:?Set BIRD_VAL_JSON to a separate validation JSON file}"
: "${BIRD_DB_DIR:?Set BIRD_DB_DIR to the directory containing db_id/db_id.sqlite}"
: "${BIRD_COLUMN_MEANING:?Set BIRD_COLUMN_MEANING to column_meaning.json}"

mkdir -p "${ROOT}/data/processed"
python3 -m agenticrl.bird_data --mode rl --data "${BIRD_TRAIN_JSON}" \
  --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
  --output "${ROOT}/data/processed/bird_rl_train.jsonl" "${@}"
python3 -m agenticrl.bird_data --mode rl --data "${BIRD_VAL_JSON}" \
  --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
  --output "${ROOT}/data/processed/bird_rl_val.jsonl" "${@}"

if [[ -n "${BIRD_SFT_TRAJECTORIES:-}" ]]; then
  : "${BIRD_SFT_EVALUATION:?Set BIRD_SFT_EVALUATION to the matching eval_results.json}"
  python3 -m agenticrl.bird_data --mode sft --data "${BIRD_TRAIN_JSON}" \
    --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
    --trajectories "${BIRD_SFT_TRAJECTORIES}" \
    --evaluation "${BIRD_SFT_EVALUATION}" \
    --output "${ROOT}/data/processed/bird_sft_train.jsonl" "${@}"
fi

if [[ -n "${BIRD_SFT_VAL_TRAJECTORIES:-}" ]]; then
  : "${BIRD_SFT_VAL_EVALUATION:?Set BIRD_SFT_VAL_EVALUATION to the matching eval_results.json}"
  python3 -m agenticrl.bird_data --mode sft --data "${BIRD_VAL_JSON}" \
    --db-dir "${BIRD_DB_DIR}" --column-meaning "${BIRD_COLUMN_MEANING}" \
    --trajectories "${BIRD_SFT_VAL_TRAJECTORIES}" \
    --evaluation "${BIRD_SFT_VAL_EVALUATION}" \
    --output "${ROOT}/data/processed/bird_sft_val.jsonl" "${@}"
fi
