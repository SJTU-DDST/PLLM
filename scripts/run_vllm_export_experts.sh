#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
export PLLM_EER_MODE=export
export PLLM_EER_CACHE_DIR="${PLLM_EER_CACHE_DIR:-/mnt/ssd-storage/pllm-experts}"
export PLLM_EER_CACHE_QUOTA_GIB="${PLLM_EER_CACHE_QUOTA_GIB:-80}"
export PYTHONPATH="${ROOT}/vllm_patch:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${PLLM_EER_CACHE_DIR}"
echo "This command loads the full model and exports transformed Marlin experts." >&2
echo "Do not run it while GPU memory is occupied. Monitor export progress with:" >&2
echo "  python scripts/eer_runtime_ctl.py status" >&2

cd "${ROOT}"
exec bash scripts/run_vllm.sh --enforce-eager "$@"
