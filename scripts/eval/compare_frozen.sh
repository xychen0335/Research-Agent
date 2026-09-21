#!/usr/bin/env bash
# Compare frozen eval directories. Missing runs stay not_run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m research_agent.cli compare \
  --runs "${RUNS:-outputs/psqa-qwen35-4b-base,outputs/psqa-no-retrieval,outputs/grpo}" \
  --output "${OUTPUT:-outputs/compare-frozen.json}" \
  "$@"
