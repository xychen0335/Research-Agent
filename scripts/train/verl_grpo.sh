#!/usr/bin/env bash
# Start verl's GRPO trainer with this project's Agent Loop and reward hook.
# Does not implement GRPO. Does not run on machines without CUDA + verl.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python -m research_agent.training.verl.launch "$@"
