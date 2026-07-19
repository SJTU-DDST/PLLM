#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PLLM_VENV_DIR:-${ROOT_DIR}/.venv}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv was not found; set UV_BIN or install uv first." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.12 "${VENV_DIR}"
fi

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
  --editable "${ROOT_DIR}[inference,test]" \
  "matplotlib>=3.9,<4"
"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/apply_vllm_hibercache_patch.py"

echo "PLLM environment is ready: ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
