from __future__ import annotations

import importlib.metadata
from typing import Any


SUPPORTED_VLLM = "0.25.1"
_INSTALLED = False


def apply_prompt_start(sampling_params: Any, vllm_xargs: dict[str, Any] | None) -> Any:
    if not vllm_xargs or "routed_experts_prompt_start" not in vllm_xargs:
        return sampling_params
    value = int(vllm_xargs["routed_experts_prompt_start"])
    if value < 0:
        raise ValueError("routed_experts_prompt_start cannot be negative")
    sampling_params.routed_experts_prompt_start = value
    return sampling_params


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    version = importlib.metadata.version("vllm")
    if version != SUPPORTED_VLLM:
        raise RuntimeError(
            f"PLLM decode trace patch requires vLLM {SUPPORTED_VLLM}, found {version}"
        )
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    original = CompletionRequest.to_sampling_params

    def to_sampling_params(request, *args, **kwargs):
        result = original(request, *args, **kwargs)
        return apply_prompt_start(result, request.vllm_xargs)

    CompletionRequest.to_sampling_params = to_sampling_params
    _INSTALLED = True
