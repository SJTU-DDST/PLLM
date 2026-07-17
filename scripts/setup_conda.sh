#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-${HOME}/anaconda3/bin/conda}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx pllm; then
  "${CONDA_BIN}" create -n pllm python=3.12 pip -y
fi

PYTHON="${HOME}/anaconda3/envs/pllm/bin/python"
"${PYTHON}" -m pip install --no-cache-dir \
  vllm==0.25.1 Flask requests psutil nvidia-ml-py PySide6 dbus-next pytest pytest-cov
"${PYTHON}" -m pip install --no-cache-dir fastsafetensors==0.3.3
"${PYTHON}" -m pip install --no-cache-dir -e "${ROOT_DIR}" --no-deps
"${PYTHON}" "${ROOT_DIR}/scripts/apply_vllm_hibercache_patch.py"
"${CONDA_BIN}" clean -a -y

echo "PLLM environment is ready: ${HOME}/anaconda3/envs/pllm"
