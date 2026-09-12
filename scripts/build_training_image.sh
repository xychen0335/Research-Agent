#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_ROOT="${ROOT}/upstream/verl"
EXPECTED_COMMIT=10db40d0da4d59150bb389960b77585f81a89b8d
IMAGE_TAG=${IMAGE_TAG:-agenticrl-verl:10db40d-cu130}

command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
[[ "$(git -C "${VERL_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}" ]] || {
  echo "verl is not at ${EXPECTED_COMMIT}" >&2
  exit 1
}
docker build -f "${VERL_ROOT}/docker/Dockerfile.uv.cu130" -t "${IMAGE_TAG}" "${VERL_ROOT}"
echo "Built ${IMAGE_TAG} from pinned verl commit ${EXPECTED_COMMIT}."
