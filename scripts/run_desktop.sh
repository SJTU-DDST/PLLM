#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HOME}/anaconda3/envs/pllm/bin/python"

cd "${ROOT_DIR}"
exec "${PYTHON}" -m pllm.desktop "$@"

