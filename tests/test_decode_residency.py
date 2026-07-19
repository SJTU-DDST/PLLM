from __future__ import annotations

import itertools
import math
import random

import pytest
import numpy as np

from pllm.decode_residency import (
    DecodeResidencyGuardrail,
    DecodeRouteWindow,
    HorizonAwareLayerPlanner,
    simulate_decode_cache,
)


def test_route_window_separates_prefill_and_decode_and_ranks_hot_experts() -> None:
    window = DecodeRouteWindow(
        [1, 3], experts_per_layer=8, window_steps=2,
        validation_profiles=(1, 3, 8),
    )
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


def test_route_window_does_not_train_on_an_unsealed_future_row() -> None:
    window = DecodeRouteWindow(
        [0], experts_per_layer=8, window_steps=2,
        validation_profiles=(3, 8),
    )
    window.set_phase("decode")
    window.observe(0, [0, 1])
    window.observe(0, [0, 2])
    window.observe(0, [3, 4])

    assert window.hot_experts(0, 3) == [0, 2, 1]
    assert window.projected_hit_rate(3) == pytest.approx(1.0)


def test_route_prediction_is_scored_only_on_the_following_window() -> None:
    window = DecodeRouteWindow(
        [0], experts_per_layer=8, window_steps=2,
        validation_profiles=(2, 8),
    )
    window.set_phase("decode")
    window.observe_rows(0, [[0, 1], [0, 1]])
    first = window.status(profiles=(2, 8))["next_window"]
    assert first["prediction_ready"] is False
    assert first["profiles"]["2"]["heldout_windows"] == 0

    window.observe_rows(0, [[0, 2], [0, 2]])
    second = window.status(profiles=(2, 8))["next_window"]
    metric = second["profiles"]["2"]["per_layer"]["0"]
    assert second["prediction_ready"] is True
    assert metric["byte_hit_rate_lower_bound"] == pytest.approx(0.5)
    assert metric["p95_misses_per_token_upper_bound"] == 1.0
    assert metric["max_misses_per_token_upper_bound"] == 1.0


def test_route_prediction_resets_at_a_new_idle_request() -> None:
    window = DecodeRouteWindow(
        [0], experts_per_layer=8, window_steps=2, validation_profiles=(2, 8)
    )
    window.set_phase("decode")
    window.observe_rows(0, [[0, 1], [0, 1], [0, 2], [0, 2]])
    assert window.status(profiles=(2,))["next_window"]["prediction_ready"] is True

    window.set_phase("prefill", reset_decode=True)

    status = window.status(profiles=(2,))["next_window"]
    assert status["prediction_ready"] is False
    assert status["minimum_completed_windows"] == 0
    assert status["request_generation"] == 1
    assert status["route_generation"] > 0
    assert window.status()["prefill_layer_rows_retained"] == 0


def test_horizon_planner_changes_only_the_predictable_layer() -> None:
    prediction = {
        "prediction_ready": True,
        "minimum_completed_windows": 2,
        "profiles": {
            "2": {
                "per_layer": {
                    "0": {
                        "byte_hit_rate_lower_bound": 0.95,
                        "mean_misses_per_token_upper_bound": 0.1,
                        "p95_misses_per_token_upper_bound": 1.0,
                        "heldout_windows": 1,
                    },
                    "1": {
                        "byte_hit_rate_lower_bound": 0.5,
                        "mean_misses_per_token_upper_bound": 1.0,
                        "p95_misses_per_token_upper_bound": 2.0,
                        "heldout_windows": 1,
                    },
                }
            }
        },
    }
    planner = HorizonAwareLayerPlanner(
        top_k=2,
        minimum_byte_hit_rate=0.95,
        maximum_slowdown_ratio=5.0,
        miss_latency_curve_ms={1: 1.0, 2: 2.0},
        reclaim_bucket_bytes=1,
    )
    result = planner.choose(
        phase="decode",
        prediction=prediction,
        candidate_slots=[2],
        layer_bytes={0: 400, 1: 400},
        current_slots_by_layer={0: 4, 1: 4},
        target_reclaim_bytes=200,
        io_budget_gib_s=1,
        token_rate=1,
        baseline_tpot_ms=100,
        remaining_decode_tokens=512,
        resize_copy_gib_s=1,
        expand_gib_s=1,
        rebuild_ms_per_layer=0,
        release_deadline_ms=100,
        full_slots=4,
    )

    assert result.action == "decode_elastic"
    assert result.slots_by_layer == {0: 2, 1: 4}
    assert result.projected_reclaim_bytes == 200
    assert result.estimated_slowdown_ratio < 5


def test_horizon_planner_rejects_unamortized_short_decode() -> None:
    metric = {
        "byte_hit_rate_lower_bound": 1.0,
        "mean_misses_per_token_upper_bound": 0.0,
        "p95_misses_per_token_upper_bound": 0.0,
        "heldout_windows": 1,
    }
    planner = HorizonAwareLayerPlanner(
        top_k=2,
        maximum_slowdown_ratio=2.0,
        miss_latency_curve_ms={1: 1.0, 2: 2.0},
        reclaim_bucket_bytes=1,
    )
    result = planner.choose(
        phase="decode",
        prediction={
            "prediction_ready": True,
            "minimum_completed_windows": 2,
            "profiles": {"2": {"per_layer": {"0": metric}}},
        },
        candidate_slots=[2],
        layer_bytes={0: 4 * 1024**3},
        current_slots_by_layer={0: 4},
        target_reclaim_bytes=2 * 1024**3,
        io_budget_gib_s=1,
        token_rate=1,
        baseline_tpot_ms=100,
        remaining_decode_tokens=2,
        resize_copy_gib_s=100,
        expand_gib_s=0.5,
        rebuild_ms_per_layer=1,
        release_deadline_ms=500,
        full_slots=4,
    )

    assert result.action == "yield"


def test_bucket_frontier_preserves_a_feasible_higher_reclaim_state() -> None:
    planner = HorizonAwareLayerPlanner(
        top_k=1,
        minimum_byte_hit_rate=0.0,
        maximum_slowdown_ratio=9.0,
        miss_latency_curve_ms={1: 1.0},
        reclaim_bucket_bytes=64,
    )
    prediction = {
        "prediction_ready": True,
        "minimum_completed_windows": 2,
        "profiles": {
            "2": {"per_layer": {"0": {
                "byte_hit_rate_lower_bound": 0.9,
                "mean_misses_per_token_upper_bound": 0.1,
                "p95_misses_per_token_upper_bound": 1.0,
                "max_misses_per_token_upper_bound": 1.0,
                "heldout_windows": 1,
            }}},
            "3": {"per_layer": {"0": {
                "byte_hit_rate_lower_bound": 1.0,
                "mean_misses_per_token_upper_bound": 0.0,
                "p95_misses_per_token_upper_bound": 0.0,
                "max_misses_per_token_upper_bound": 0.0,
                "heldout_windows": 1,
            }}},
        },
    }

    result = planner.choose(
        phase="decode",
        prediction=prediction,
        candidate_slots=[2, 3],
        layer_bytes={0: 100},
        current_slots_by_layer={0: 4},
        target_reclaim_bytes=40,
        io_budget_gib_s=1,
        token_rate=1,
        baseline_tpot_ms=100,
        remaining_decode_tokens=1000,
        resize_copy_gib_s=1,
        expand_gib_s=1,
        rebuild_ms_per_layer=0,
        release_deadline_ms=100,
        full_slots=4,
    )

    assert result.action == "decode_elastic"
    assert result.slots_by_layer == {0: 2}
    assert result.projected_reclaim_bytes == 50


def test_release_deadline_includes_destructive_immediate_expansion() -> None:
    metric = {
        "byte_hit_rate_lower_bound": 1.0,
        "mean_misses_per_token_upper_bound": 0.0,
        "p95_misses_per_token_upper_bound": 0.0,
        "max_misses_per_token_upper_bound": 0.0,
        "heldout_windows": 1,
    }
    planner = HorizonAwareLayerPlanner(
        top_k=1,
        minimum_byte_hit_rate=0.0,
        maximum_slowdown_ratio=9.0,
        miss_latency_curve_ms={1: 1.0},
        reclaim_bucket_bytes=1,
    )
    result = planner.choose(
        phase="decode",
        prediction={
            "prediction_ready": True,
            "minimum_completed_windows": 2,
            "profiles": {"3": {"per_layer": {"0": metric}}},
        },
        candidate_slots=[3],
        layer_bytes={0: 1024**3},
        current_slots_by_layer={0: 2},
        target_reclaim_bytes=256 * 1024**2,
        io_budget_gib_s=1,
        token_rate=1,
        baseline_tpot_ms=100,
        remaining_decode_tokens=1000,
        resize_copy_gib_s=100,
        expand_gib_s=1,
        rebuild_ms_per_layer=0,
        release_deadline_ms=500,
        full_slots=4,
    )

    assert result.action == "yield"


def test_pareto_bucket_planner_matches_small_exhaustive_oracle() -> None:
    rng = random.Random(7)
    gib = 1024**3
    for _ in range(1000):
        layer_bytes = {
            layer: rng.randint(1 * gib, 4 * gib) for layer in range(3)
        }
        maximum = sum(round(size * 2 / 4) for size in layer_bytes.values())
        target = rng.randint(1, max(1, maximum))
        means = {
            (layer, 2): rng.choice((0.0, 0.05, 0.2, 0.5))
            for layer in layer_bytes
        }
        for layer in layer_bytes:
            means[(layer, 3)] = min(
                means[(layer, 2)], rng.choice((0.0, 0.02, 0.1))
            )
        prediction = {
            "prediction_ready": True,
            "minimum_completed_windows": 2,
            "profiles": {
                str(slots): {
                    "per_layer": {
                        str(layer): {
                            "byte_hit_rate_lower_bound": 1.0 - means[(layer, slots)] / 2,
                            "mean_misses_per_token_upper_bound": means[(layer, slots)],
                            "p95_misses_per_token_upper_bound": float(
                                math.ceil(means[(layer, slots)])
                            ),
                            "max_misses_per_token_upper_bound": float(
                                math.ceil(means[(layer, slots)] * 2)
                            ),
                            "heldout_windows": 1,
                        }
                        for layer in layer_bytes
                    }
                }
                for slots in (2, 3)
            },
        }
        minimum_hit = rng.uniform(0.8, 0.99)
        maximum_slowdown = rng.uniform(2.0, 8.0)
        io_budget = rng.uniform(0.1, 2.0)
        token_rate = rng.uniform(1.0, 10.0)
        baseline_tpot = rng.uniform(50.0, 200.0)
        horizon = rng.choice((128, 256, 512, 1024))
        copy_gib_s = rng.uniform(20.0, 100.0)
        expand_gib_s = rng.uniform(0.5, 3.0)
        rebuild_ms = rng.uniform(0.0, 10.0)
        deadline_ms = rng.uniform(100.0, 1000.0)
        current = {layer: rng.choice((2, 3, 4)) for layer in layer_bytes}
        planner = HorizonAwareLayerPlanner(
            top_k=2,
            minimum_byte_hit_rate=minimum_hit,
            maximum_slowdown_ratio=maximum_slowdown,
            miss_latency_curve_ms={1: 1.0, 2: 2.0},
            reclaim_bucket_bytes=64 * 1024**2,
        )
        result = planner.choose(
            phase="decode",
            prediction=prediction,
            candidate_slots=[2, 3],
            layer_bytes=layer_bytes,
            current_slots_by_layer=current,
            target_reclaim_bytes=target,
            io_budget_gib_s=io_budget,
            token_rate=token_rate,
            baseline_tpot_ms=baseline_tpot,
            remaining_decode_tokens=horizon,
            resize_copy_gib_s=copy_gib_s,
            expand_gib_s=expand_gib_s,
            rebuild_ms_per_layer=rebuild_ms,
            release_deadline_ms=deadline_ms,
            full_slots=4,
        )

        candidates = []
        for slots_tuple in itertools.product((2, 3, 4), repeat=3):
            reclaim = sum(
                round(layer_bytes[layer] * (4 - slots) / 4)
                for layer, slots in enumerate(slots_tuple)
            )
            if reclaim < target:
                continue
            mean_misses = sum(
                means.get((layer, slots), 0.0)
                for layer, slots in enumerate(slots_tuple)
            )
            miss_bytes = sum(
                means.get((layer, slots), 0.0) * layer_bytes[layer] / 4
                for layer, slots in enumerate(slots_tuple)
            )
            total_access_bytes = sum(size / 4 * 2 for size in layer_bytes.values())
            hit_rate = 1.0 - miss_bytes / total_access_bytes
            miss_gib_s = miss_bytes / gib * token_rate
            risk_ms = sum(
                math.ceil(means.get((layer, slots), 0.0) * 2)
                for layer, slots in enumerate(slots_tuple)
            )
            immediate = 0.0
            future = 0.0
            for layer, slots in enumerate(slots_tuple):
                size = layer_bytes[layer]
                changed = slots != current[layer]
                shrink = round(size * slots / 4) if changed and slots < current[layer] else 0
                expand = round(size * slots / 4) if changed and slots > current[layer] else 0
                immediate += shrink / (copy_gib_s * gib)
                immediate += expand / (expand_gib_s * gib)
                immediate += int(changed) * rebuild_ms / 1000
                if slots < 4:
                    future += size / (expand_gib_s * gib) + rebuild_ms / 1000
            transition = immediate + future
            slowdown = (
                baseline_tpot + risk_ms + transition * 1000 / horizon
            ) / baseline_tpot
            if (
                hit_rate < minimum_hit
                or miss_gib_s > io_budget
                or immediate * 1000 > deadline_ms
                or slowdown >= maximum_slowdown
            ):
                continue
            candidates.append((slowdown, reclaim - target, reclaim))
        if not candidates:
            assert result.action == "yield"
        else:
            expected = min(candidates)
            assert result.action == "decode_elastic"
            assert result.projected_reclaim_bytes == expected[2]
            assert result.estimated_slowdown_ratio == pytest.approx(expected[0])


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
