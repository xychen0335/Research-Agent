#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_FILE=${1:?Usage: scripts/evaluate_bird_run.sh RUN_JSONL [OUTPUT_JSON]}
OUTPUT_FILE=${2:-${RUN_FILE%.jsonl}.bird-eval.json}
TRAJECTORY_FILE=${OUTPUT_FILE%.json}.trajectories.jsonl
GOLD_FILE=${BIRD_GOLD_FILE:-${ROOT}/data/splits/frozen_test.json}
DB_DIR=${BIRD_DB_DIR:-${ROOT}/data/databases/mini_dev}
PYTHON_BIN=${ROOT}/upstream/verl/.venv/bin/python

[[ -f "${RUN_FILE}" ]] || { echo "Missing rollout file: ${RUN_FILE}" >&2; exit 1; }
[[ -f "${GOLD_FILE}" ]] || { echo "Missing gold file: ${GOLD_FILE}" >&2; exit 1; }
[[ -d "${DB_DIR}" ]] || { echo "Missing database directory: ${DB_DIR}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python3

bash "${ROOT}/scripts/apply_upstream_patches.sh"
"${PYTHON_BIN}" "${ROOT}/scripts/export_bird_trajectories.py" \
  --input "${RUN_FILE}" \
  --output "${TRAJECTORY_FILE}" \
  --expected-records 500
"${PYTHON_BIN}" "${ROOT}/upstream/BIRD-RL/bird_rl/inference/bird/evaluate.py" \
  --trajectory "${TRAJECTORY_FILE}" \
  --gold "${GOLD_FILE}" \
  --db-dir "${DB_DIR}" \
  --output "${OUTPUT_FILE}" \
  --threads "${BIRD_EVAL_THREADS:-8}" \
  --timeout "${BIRD_SQL_TIMEOUT:-30}"
