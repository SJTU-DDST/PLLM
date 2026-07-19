from __future__ import annotations

from types import SimpleNamespace

import pytest

from pllm.vllm_decode_trace_patch import apply_prompt_start


def test_prompt_start_is_opt_in() -> None:
    params = SimpleNamespace(routed_experts_prompt_start=0)
    assert apply_prompt_start(params, None) is params
    assert params.routed_experts_prompt_start == 0


def test_prompt_start_flows_from_vllm_xargs() -> None:
    params = SimpleNamespace(routed_experts_prompt_start=0)
    apply_prompt_start(params, {"routed_experts_prompt_start": 384})
    assert params.routed_experts_prompt_start == 384


def test_prompt_start_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        apply_prompt_start(
            SimpleNamespace(routed_experts_prompt_start=0),
            {"routed_experts_prompt_start": -1},
        )
