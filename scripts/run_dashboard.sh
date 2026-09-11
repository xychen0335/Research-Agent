#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STREAMLIT_BIN=${STREAMLIT_BIN:-}
if [[ -z "${STREAMLIT_BIN}" ]] && command -v streamlit >/dev/null; then
  STREAMLIT_BIN=$(command -v streamlit)
elif [[ -z "${STREAMLIT_BIN}" ]] && [[ -x "${ROOT}/upstream/verl/.venv/bin/streamlit" ]]; then
  STREAMLIT_BIN="${ROOT}/upstream/verl/.venv/bin/streamlit"
fi
[[ -n "${STREAMLIT_BIN}" ]] || { echo "Install requirements-ui.txt first." >&2; exit 1; }
cd "${ROOT}"
exec "${STREAMLIT_BIN}" run app.py "${@}"
