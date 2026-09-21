#!/usr/bin/env bash
# Install torch / vLLM / verl on a CUDA node. Clones verl into upstream/verl
# (same layout as the official docs: git clone, then pip install -e).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p upstream
if [ ! -d upstream/verl/.git ]; then
  git clone https://github.com/verl-project/verl.git upstream/verl
fi
python -m pip install -r requirements-train.txt
python -m pip install -e "upstream/verl"
python scripts/setup/check_env.py
