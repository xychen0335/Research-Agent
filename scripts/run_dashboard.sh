#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STREAMLIT_BIN=${STREAMLIT_BIN:-}
if [[ -z "${STREAMLIT_BIN}" ]] && [[ -x "${ROOT}/.venv-ui/bin/streamlit" ]]; then
  STREAMLIT_BIN="${ROOT}/.venv-ui/bin/streamlit"
elif [[ -z "${STREAMLIT_BIN}" ]] && command -v streamlit >/dev/null; then
  STREAMLIT_BIN=$(command -v streamlit)
fi
[[ -n "${STREAMLIT_BIN}" ]] || { echo "Run scripts/setup_ui.sh first." >&2; exit 1; }
cd "${ROOT}"
exec "${STREAMLIT_BIN}" run app.py \
  --server.address "${STREAMLIT_ADDRESS:-127.0.0.1}" \
  "${@}"
