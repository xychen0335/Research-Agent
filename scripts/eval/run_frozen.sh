#!/usr/bin/env bash
# Frozen 8-task Base probe. Requires a live policy; do not run alongside another Ollama job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m research_agent.cli eval \
  --data data/processed/papersearchqa-dev \
  --eval-config configs/evaluation/base_mini.yaml \
  --baseline agent \
  --policy "${POLICY:-ollama}" \
  --model-name "${MODEL_NAME:-qwen3.5:4b}" \
  --run-id "${RUN_ID:-frozen-qwen35-4b-base}" \
  --output "${OUTPUT:-outputs/frozen-qwen35-4b-base}" \
  ${ADAPTER:+--adapter "$ADAPTER"} \
  ${LOCAL_FILES_ONLY:+--local-files-only} \
  "$@"
