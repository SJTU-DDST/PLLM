#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "Conda was not found; set CONDA_EXE to its executable path." >&2
  exit 1
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx pllm; then
  "${CONDA_BIN}" create -n pllm python=3.12 pip -y
fi

"${CONDA_BIN}" run -n pllm python -m pip install --no-cache-dir \
  vllm==0.25.1 Flask requests psutil nvidia-ml-py PySide6 dbus-next pytest pytest-cov
"${CONDA_BIN}" run -n pllm python -m pip install --no-cache-dir fastsafetensors==0.3.3
"${CONDA_BIN}" run -n pllm python -m pip install --no-cache-dir -e "${ROOT_DIR}" --no-deps
"${CONDA_BIN}" run -n pllm python "${ROOT_DIR}/scripts/apply_vllm_hibercache_patch.py"
"${CONDA_BIN}" clean -a -y

echo "PLLM environment 'pllm' is ready. Activate it with: conda activate pllm"
