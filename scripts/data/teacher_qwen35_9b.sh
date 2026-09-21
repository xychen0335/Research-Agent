#!/usr/bin/env bash
# Teacher sampling with Qwen3.5-9B. Does not read gold labels.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python scripts/data/teacher.py \
  --data data/processed/papersearchqa-dev \
  --policy ollama \
  --model-name qwen3.5:9b \
  --samples 2 \
  --stop-when-kept \
  --output outputs/teacher-qwen35-9b \
  "$@"
