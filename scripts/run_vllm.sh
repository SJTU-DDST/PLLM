#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
VLLM_BIN="${HOME}/anaconda3/envs/pllm/bin/vllm"

if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
  echo "Model is not readable: ${MODEL_PATH}" >&2
  exit 1
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_FASTSAFETENSORS_QUEUE_SIZE="${VLLM_FASTSAFETENSORS_QUEUE_SIZE:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

if [[ -n "${PLLM_EER_RDMA_POOL_INDEX:-}" ]]; then
  if [[ ! -r "${PLLM_EER_RDMA_POOL_INDEX}" ]]; then
    echo "RDMA warm-profile index is not readable: ${PLLM_EER_RDMA_POOL_INDEX}" >&2
    exit 1
  fi
  export PLLM_EER_RDMA_POOL_BINARY="${PLLM_EER_RDMA_POOL_BINARY:-${PWD}/rdma_bridge/build/pllm-rdma-pool}"
fi

HIBERCACHE_DIR="${HIBERCACHE_DIR:-/mnt/ssd-storage/pllm-cache}"
mkdir -p "${HIBERCACHE_DIR}"
KV_TRANSFER_CONFIG="$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":536870912,"eviction_policy":"arc","secondary_tiers":[{"type":"fs","root_dir":"%s","n_read_threads":8,"n_write_threads":4}]}}' "${HIBERCACHE_DIR}")"
SLEEP_MODE_ARGS=()
if [[ "${PLLM_VLLM_ENABLE_SLEEP_MODE:-1}" == "1" ]]; then
  SLEEP_MODE_ARGS+=(--enable-sleep-mode)
fi
HIBERCACHE_ARGS=()
if [[ "${PLLM_VLLM_ENABLE_HIBERCACHE:-1}" == "1" ]]; then
  HIBERCACHE_ARGS+=(
    --enable-prefix-caching
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
  )
fi

if ! "${HOME}/anaconda3/envs/pllm/bin/python" \
  "${PWD}/scripts/apply_vllm_hibercache_patch.py" --check >/dev/null 2>&1; then
  echo "Warning: HiberCache patch is not installed; mode=keep uses token recompute fallback." >&2
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name nvidia/nemotron-3-super \
  --host 127.0.0.1 \
  --port "${VLLM_PORT:-8000}" \
  "${SLEEP_MODE_ARGS[@]}" \
  --load-format "${PLLM_VLLM_LOAD_FORMAT:-fastsafetensors}" \
  "${HIBERCACHE_ARGS[@]}" \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization "${PLLM_VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --enable-chunked-prefill \
  --max-num-seqs 2 \
  --max-num-batched-tokens "${PLLM_VLLM_MAX_BATCHED_TOKENS:-8192}" \
  --max-model-len 32768 \
  --linear-backend "${PLLM_VLLM_LINEAR_BACKEND:-cutlass}" \
  --moe-backend marlin \
  --mamba-ssm-cache-dtype float16 \
  --quantization modelopt \
  --reasoning-parser-plugin "${MODEL_PATH}/super_v3_reasoning_parser.py" \
  --reasoning-parser super_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
