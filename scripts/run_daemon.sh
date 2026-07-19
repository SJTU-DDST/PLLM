#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PLLM_PYTHON:-}" ]]; then
  PYTHON="${PLLM_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/anaconda3/envs/pllm/bin/python" ]]; then
  PYTHON="${HOME}/anaconda3/envs/pllm/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Python was not found; set PLLM_PYTHON or activate the PLLM environment." >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" -m pllm.daemon "$@"
