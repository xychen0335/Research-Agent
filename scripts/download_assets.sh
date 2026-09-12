#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_ROOT="${ROOT}/upstream/verl"
RAW="${ROOT}/data/raw"
ARCHIVES="${RAW}/archives"
EXTRACTED="${RAW}/extracted"
mkdir -p "${ARCHIVES}" "${EXTRACTED}/train" "${EXTRACTED}/mini_dev"

[[ -f "${VERL_ROOT}/uv.lock" ]] || { echo "Run scripts/setup_autodl.sh first." >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }
cd "${VERL_ROOT}"
uv run --python 3.12 --frozen --all-packages --extra vllm --extra fsdp \
  python3 "${ROOT}/scripts/download_hf_assets.py" --model

TRAIN_ZIP="${ARCHIVES}/bird_train.zip"
MINI_ZIP="${ARCHIVES}/bird_mini_dev.zip"
if [[ ! -s "${TRAIN_ZIP}" ]]; then
  curl --fail --location --retry 3 \
    https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip -o "${TRAIN_ZIP}"
fi
if [[ ! -s "${MINI_ZIP}" ]]; then
  uvx --from gdown==5.2.0 gdown 13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG -O "${MINI_ZIP}"
fi

unzip -q -o "${TRAIN_ZIP}" -d "${EXTRACTED}/train"
unzip -q -o "${MINI_ZIP}" -d "${EXTRACTED}/mini_dev"
PYTHON="${VERL_ROOT}/.venv/bin/python3"
"${PYTHON}" "${ROOT}/scripts/normalize_bird_databases.py" \
  --extracted "${EXTRACTED}/train" \
  --records "${RAW}/bird23-train-filtered/train.jsonl" \
  --destination "${ROOT}/data/databases/train"
"${PYTHON}" "${ROOT}/scripts/normalize_bird_databases.py" \
  --extracted "${EXTRACTED}/mini_dev" \
  --records "${RAW}/bird-mini-dev/mini_dev_sqlite.json" \
  --destination "${ROOT}/data/databases/mini_dev"
"${PYTHON}" "${ROOT}/scripts/build_data_splits.py"

cd "${ROOT}"
if [[ -f data/SHA256SUMS ]]; then
  sha256sum --check data/SHA256SUMS
else
  sha256sum \
    data/raw/bird23-train-filtered/train.jsonl \
    data/raw/bird23-train-filtered/column_meaning.json \
    data/raw/bird-mini-dev/mini_dev_sqlite.json \
    data/raw/archives/bird_train.zip \
    data/raw/archives/bird_mini_dev.zip > data/SHA256SUMS
fi
echo "Pinned data downloaded, normalized, split, and recorded in data/SHA256SUMS. Commit this manifest."
