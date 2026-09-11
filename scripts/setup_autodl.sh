#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAMS="${ROOT}/upstream"
BIRD_COMMIT=b26c4285ef69dfb6c096b076a7499018c6b370ca
VERL_COMMIT=10db40d0da4d59150bb389960b77585f81a89b8d

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "Run this script on the AutoDL NVIDIA instance." >&2; exit 1; }
mkdir -p "${UPSTREAMS}" "${ROOT}/outputs/setup"

clone_at_commit() {
  local url=$1
  local directory=$2
  local commit=$3
  if [[ ! -d "${directory}/.git" ]]; then
    git clone "${url}" "${directory}"
  fi
  git -C "${directory}" fetch origin "${commit}" --depth 1
  git -C "${directory}" checkout --detach "${commit}"
  test "$(git -C "${directory}" rev-parse HEAD)" = "${commit}"
}

clone_at_commit https://github.com/bird-bench/BIRD-RL.git "${UPSTREAMS}/BIRD-RL" "${BIRD_COMMIT}"
clone_at_commit https://github.com/verl-project/verl.git "${UPSTREAMS}/verl" "${VERL_COMMIT}"

if ! command -v uv >/dev/null; then
  python3 -m pip install --user uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${UPSTREAMS}/verl"
uv sync --frozen --all-packages --extra vllm --extra fsdp
uv pip install func-timeout
uv pip install -r "${ROOT}/requirements-ui.txt"

nvidia-smi > "${ROOT}/outputs/setup/nvidia-smi.txt"
uv run python3 -c 'import json, torch, transformers, vllm; print(json.dumps({"torch": torch.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda}))' > "${ROOT}/outputs/setup/versions.json"
echo "Environment ready. Recorded versions in ${ROOT}/outputs/setup/."
