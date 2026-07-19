from __future__ import annotations

import pytest
import numpy as np

from pllm.decode_residency import (
    DecodeResidencyGuardrail,
    DecodeRouteWindow,
    simulate_decode_cache,
)


def test_route_window_separates_prefill_and_decode_and_ranks_hot_experts() -> None:
    window = DecodeRouteWindow([1, 3], experts_per_layer=8, window_steps=2)
    window.set_phase("prefill")
    window.observe(1, [0, 1, 2], token_count=16)
    window.set_phase("decode", reset_decode=True)
    window.observe(1, [4, 5], token_count=1)
    window.observe(1, [4, 6], token_count=1)

    assert window.hot_experts(1, 3) == [4, 6, 5]
    assert window.projected_hit_rate(1) == pytest.approx(0.5)
    status = window.status(profiles=(1, 3, 8))
    assert status["phase"] == "decode"
    assert status["decode_observations"] == 2
    assert status["gpu_persistent_bytes"] == 0
    assert status["projected_byte_hit_rate"]["8"] == 1.0


def test_route_window_expires_old_decode_rows() -> None:
    window = DecodeRouteWindow([0], experts_per_layer=8, window_steps=2)
    window.set_phase("decode")
    window.observe(0, [0, 1])
    window.observe(0, [0, 2])
    window.observe(0, [3, 4])

    assert window.hot_experts(0, 3) == [3, 4, 0]
    assert window.projected_hit_rate(3) == pytest.approx(0.75)


def test_guardrail_forbids_prefill_eviction() -> None:
    guardrail = DecodeResidencyGuardrail(2.538)
    decision = guardrail.choose(
        "prefill", {384: 0.99}, [384], 2.0, 5.0, 50.0
    )
    assert decision.action == "full_resident"
    assert decision.slots_per_layer == 512


def test_guardrail_selects_smallest_safe_decode_profile() -> None:
    guardrail = DecodeResidencyGuardrail(
        2.538, minimum_byte_hit_rate=0.95, maximum_slowdown_ratio=5.0
    )
    decision = guardrail.choose(
        "decode",
        {320: 0.90, 384: 0.96, 448: 0.99},
        [320, 384, 448],
        io_budget_gib_s=2.0,
        token_rate=5.0,
        baseline_tpot_ms=100.0,
    )
    assert decision.action == "decode_elastic"
    assert decision.slots_per_layer == 384
    assert decision.estimated_slowdown_ratio < 5.0
    assert decision.misses_per_token == pytest.approx(35.2)
    assert decision.miss_latency_ms_per_token == pytest.approx(264.0)


def test_guardrail_yields_before_order_of_magnitude_slowdown() -> None:
    guardrail = DecodeResidencyGuardrail(
        2.538, minimum_byte_hit_rate=0.95, maximum_slowdown_ratio=3.0
    )
    decision = guardrail.choose(
        "decode",
        {384: 0.70, 448: 0.90},
        [384, 448],
        io_budget_gib_s=2.0,
        token_rate=5.0,
        baseline_tpot_ms=50.0,
    )
    assert decision.action == "yield"
    assert decision.estimated_slowdown_ratio == float("inf")


def test_decode_cache_simulation_seeds_from_prefill_and_preserves_routes() -> None:
    prefill = np.array(
        [
            [[0, 1], [2, 3]],
            [[0, 2], [2, 4]],
        ],
        dtype=np.uint16,
    )
    decode = np.array(
        [
            [[0, 2], [2, 4]],
            [[0, 3], [2, 5]],
        ],
        dtype=np.uint16,
    )

    full = simulate_decode_cache(
        prefill, decode, 8, 100, policy="lru", experts_per_layer=8
    )
    small = simulate_decode_cache(
        prefill, decode, 2, 100, policy="window_lfu", experts_per_layer=8
    )

    assert full.byte_hit_rate == 1.0
    assert small.exact_route_preserved is True
    assert small.expert_accesses == 8
    assert small.blocking_misses == 2
    assert small.miss_bytes == 200
