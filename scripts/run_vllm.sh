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
export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

HIBERCACHE_DIR="${HIBERCACHE_DIR:-/mnt/ssd-storage/pllm-cache}"
mkdir -p "${HIBERCACHE_DIR}"
KV_TRANSFER_CONFIG="$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":536870912,"eviction_policy":"arc","secondary_tiers":[{"type":"fs","root_dir":"%s","n_read_threads":8,"n_write_threads":4}]}}' "${HIBERCACHE_DIR}")"

if ! "${HOME}/anaconda3/envs/pllm/bin/python" \
  "${PWD}/scripts/apply_vllm_hibercache_patch.py" --check >/dev/null 2>&1; then
  echo "Warning: HiberCache patch is not installed; mode=keep uses token recompute fallback." >&2
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name nvidia/nemotron-3-super \
  --host 127.0.0.1 \
  --port "${VLLM_PORT:-8000}" \
  --enable-sleep-mode \
  --load-format "${VLLM_LOAD_FORMAT:-fastsafetensors}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --max-num-seqs 2 \
  --max-model-len 32768 \
  --moe-backend marlin \
  --mamba-ssm-cache-dtype float16 \
  --quantization modelopt \
  --reasoning-parser-plugin "${MODEL_PATH}/super_v3_reasoning_parser.py" \
  --reasoning-parser super_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
