#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAMS="${ROOT}/upstream"
BIRD_COMMIT=b26c4285ef69dfb6c096b076a7499018c6b370ca
VERL_COMMIT=10db40d0da4d59150bb389960b77585f81a89b8d

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "Run this script on the AutoDL NVIDIA instance." >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 1; }
[[ "$(uname -s)" = Linux && "$(uname -m)" = x86_64 ]] || {
  echo "The pinned environment requires Linux x86_64." >&2
  exit 1
}
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID}" = ubuntu && "${VERSION_ID}" = 24.04 ]] || {
  echo "The pinned environment requires Ubuntu 24.04." >&2
  exit 1
}
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n 1)
IFS=, read -r GPU_NAME GPU_MEMORY GPU_DRIVER <<< "${GPU_INFO}"
[[ "${GPU_NAME}" = *A100* ]] || { echo "An NVIDIA A100 is required." >&2; exit 1; }
(( ${GPU_MEMORY//[[:space:]]/} >= 79000 )) || { echo "An A100 80GB is required." >&2; exit 1; }
(( ${GPU_DRIVER%%.*} >= 580 )) || { echo "NVIDIA driver 580 or newer is required." >&2; exit 1; }
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
bash "${ROOT}/scripts/apply_upstream_patches.sh"

UV_VERSION=0.11.16
if ! command -v uv >/dev/null || [[ "$(uv --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
  python3 -m pip install --user "uv==${UV_VERSION}"
  export PATH="${HOME}/.local/bin:${PATH}"
fi
[[ "$(uv --version | awk '{print $2}')" = "${UV_VERSION}" ]] || {
  echo "uv ${UV_VERSION} is required, found $(uv --version)." >&2
  exit 1
}

cd "${UPSTREAMS}/verl"
test "$(sha256sum uv.lock | awk '{print $1}')" = d30f7e35c9f077c3c27aae1012711a753939db81a9a2cf4bcdbdf03fab59d0a9
uv python install 3.12
uv sync --python 3.12 --frozen --all-packages --extra vllm --extra fsdp

nvidia-smi > "${ROOT}/outputs/setup/nvidia-smi.txt"
uv run --python 3.12 --frozen --all-packages --extra vllm --extra fsdp \
  python3 "${ROOT}/scripts/check_environment.py" --strict > "${ROOT}/outputs/setup/versions.json"
echo "Environment ready. Recorded versions in ${ROOT}/outputs/setup/."
