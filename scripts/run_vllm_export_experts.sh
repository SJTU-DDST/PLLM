#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_USER="$(id -un)"
export MODEL_PATH="${MODEL_PATH:-/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
export PLLM_EER_MODE=export
export PLLM_VLLM_ENABLE_SLEEP_MODE=0
export PLLM_VLLM_ENABLE_HIBERCACHE=0
export PLLM_EER_CACHE_DIR="${PLLM_EER_CACHE_DIR:-/mnt/ssd-storage/${CURRENT_USER}/pllm-experts}"
export PLLM_EER_CACHE_QUOTA_GIB="${PLLM_EER_CACHE_QUOTA_GIB:-80}"
export PYTHONPATH="${ROOT}/vllm_patch:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${PLLM_EER_CACHE_DIR}"
echo "This command loads the full model and exports transformed Marlin experts." >&2
echo "Do not run it while GPU memory is occupied. Monitor export progress with:" >&2
echo "  python scripts/eer_runtime_ctl.py status" >&2

cd "${ROOT}"
MANIFEST="${PLLM_EER_CACHE_DIR}/runtime-manifest.json"

manifest_complete() {
  "${PLLM_PYTHON:-python3}" -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("complete") else 1)' "${MANIFEST}" 2>/dev/null
}

setsid bash scripts/run_vllm.sh --enforce-eager "$@" &
VLLM_PID=$!
cleanup() {
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    return
  fi
  kill -TERM -- "-${VLLM_PID}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      return
    fi
    sleep 0.25
  done
  kill -KILL -- "-${VLLM_PID}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

while kill -0 "${VLLM_PID}" 2>/dev/null; do
  if [[ -s "${MANIFEST}" ]] && manifest_complete; then
    echo "Runtime expert export complete; stopping the export-only vLLM." >&2
    cleanup
    wait "${VLLM_PID}" 2>/dev/null || true
    trap - INT TERM EXIT
    exit 0
  fi
  sleep 1
done

set +e
wait "${VLLM_PID}"
STATUS=$?
set -e
trap - INT TERM EXIT
if [[ -s "${MANIFEST}" ]] && manifest_complete; then
  exit 0
fi
exit "${STATUS}"
