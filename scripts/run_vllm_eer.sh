#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_USER="${USER:-$(id -un)}"
MODEL_PATH="${MODEL_PATH:-/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
PLLM_EER_CACHE_DIR="${PLLM_EER_CACHE_DIR:-/mnt/ssd-storage/${CURRENT_USER}/pllm-experts}"

if [[ ! -r "${PLLM_EER_CACHE_DIR}/runtime-manifest.json" && -z "${PLLM_EER_RDMA_PEER:-}" ]]; then
  echo "No complete local runtime-expert cache and no RDMA warm source configured." >&2
  echo "Run scripts/run_vllm_export_experts.sh after the GPU becomes available." >&2
  exit 1
fi

export MODEL_PATH
export PLLM_EER_MODE=elastic
export PLLM_EER_CACHE_DIR
export PLLM_EER_SLOTS_PER_LAYER="${PLLM_EER_SLOTS_PER_LAYER:-512}"
export PLLM_VLLM_GPU_MEMORY_UTILIZATION="${PLLM_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export PLLM_VLLM_MAX_BATCHED_TOKENS="${PLLM_VLLM_MAX_BATCHED_TOKENS:-8192}"
export PLLM_VLLM_ENABLE_SLEEP_MODE="${PLLM_VLLM_ENABLE_SLEEP_MODE:-0}"
export PLLM_VLLM_ENABLE_HIBERCACHE="${PLLM_VLLM_ENABLE_HIBERCACHE:-0}"
export PLLM_EER_CACHE_QUOTA_GIB="${PLLM_EER_CACHE_QUOTA_GIB:-80}"
export PLLM_EER_RDMA_PORT="${PLLM_EER_RDMA_PORT:-17900}"
export PLLM_EER_RDMA_BINARY="${PLLM_EER_RDMA_BINARY:-${ROOT}/rdma_bridge/build/pllm-rdma-store}"
export PLLM_EER_RDMA_TOKEN_FILE="${PLLM_EER_RDMA_TOKEN_FILE:-${HOME}/.config/pllm/rdma-token}"
export PLLM_EER_RDMA_ALLOCATOR="${PLLM_EER_RDMA_ALLOCATOR:-cuda-host}"
export PLLM_EER_RDMA_IB_PORT="${PLLM_EER_RDMA_IB_PORT:-1}"
export PLLM_EER_RDMA_GID_INDEX="${PLLM_EER_RDMA_GID_INDEX:-0}"
export PLLM_VLLM_LOAD_FORMAT=safetensors
export PYTHONPATH="${ROOT}/vllm_patch:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
exec bash scripts/run_vllm.sh --enforce-eager "$@"
