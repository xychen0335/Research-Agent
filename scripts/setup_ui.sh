#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
cd "${ROOT}"
uv python install 3.12
uv venv --python 3.12 .venv-ui
uv pip install --python .venv-ui/bin/python -r requirements-ui.txt
"${ROOT}/.venv-ui/bin/python" -c 'import pandas, streamlit; print("streamlit=" + streamlit.__version__ + " pandas=" + pandas.__version__)'
