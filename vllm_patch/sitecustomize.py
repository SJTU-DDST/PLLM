"""Opt-in PLLM EER bootstrap loaded through PYTHONPATH by run_vllm_eer.sh."""

import os


if os.getenv("PLLM_EER_MODE", "off").lower() != "off":
    from pllm.vllm_eer_runtime import install

    install()

if os.getenv("PLLM_DECODE_TRACE", "0") == "1":
    from pllm.vllm_decode_trace_patch import install as install_decode_trace

    install_decode_trace()
