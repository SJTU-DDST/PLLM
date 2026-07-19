from __future__ import annotations

import base64
import io

import numpy as np

from scripts.benchmark_decode_routes import (
    decode_routes,
    heldout_layer_plan_summary,
    heldout_next_window_summary,
    split_returned_routes,
    summarize,
)


def test_decode_routes_rejects_non_tensor_layout() -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((4, 3, 2), dtype=np.uint16), allow_pickle=False)

    decoded = decode_routes(base64.b64encode(buffer.getvalue()).decode())

    assert decoded.shape == (4, 3, 2)
    assert decoded.dtype == np.uint16


def test_split_routes_uses_vllm_prompt_then_completion_minus_one_layout() -> None:
    routes = np.zeros((7, 2, 2), dtype=np.uint16)

    prompt, decode = split_returned_routes(
        routes,
        prompt_tokens=10,
        prompt_start=6,
        completion_tokens=4,
    )

    assert prompt.shape[0] == 4
    assert decode.shape[0] == 3


def test_heldout_summary_never_scores_a_window_on_its_own_ranking() -> None:
    routes = np.array(
        [
            [[0, 1]],
            [[0, 1]],
            [[0, 2]],
            [[0, 2]],
        ],
        dtype=np.uint16,
    )

    result = heldout_next_window_summary(
        [routes],
        [2],
        experts_per_layer=4,
        expert_bytes=100,
        window_tokens=2,
        miss_curve_ms={1: 1.0, 2: 2.0},
    )

    profile = result["profiles"]["2"]
    assert profile["heldout_transitions"] == 1
    assert profile["byte_hit_rate"] == 0.5
    assert profile["blocking_misses"] == 2
    assert profile["total_stall_ms_p95"] == 1.0


def test_heldout_summary_never_forms_windows_across_request_boundaries() -> None:
    first = np.array([[[0, 1]], [[0, 1]]], dtype=np.uint16)
    second = np.array([[[0, 2]], [[0, 2]]], dtype=np.uint16)

    result = heldout_next_window_summary(
        [first, second],
        [2],
        experts_per_layer=4,
        expert_bytes=100,
        window_tokens=2,
    )

    assert result["available"] is False
    assert result["reason"] == "no request contains two complete held-out windows"


def test_heldout_layer_plan_reports_token_total_tail() -> None:
    routes = np.array(
        [
            [[0, 1], [2, 3]],
            [[0, 1], [2, 3]],
            [[0, 2], [2, 4]],
            [[0, 2], [2, 4]],
        ],
        dtype=np.uint16,
    )

    result = heldout_layer_plan_summary(
        [routes],
        {0: 2, 1: 6},
        experts_per_layer=6,
        expert_bytes=100,
        window_tokens=2,
        miss_curve_ms={1: 1.0, 2: 2.0},
    )

    assert result["blocking_misses"] == 2
    assert result["total_stall_ms_p95"] == 1.0
    assert result["total_stall_ms_cvar95"] == 1.0


def test_summary_aggregates_simulated_decode_profiles() -> None:
    rows = [
        {
            "status": "ok",
            "f1": 0.5,
            "latency_seconds": 2.0,
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "captured_decode_tokens": 2,
            "route_file_bytes": 100,
            "simulations": [
                {
                    "policy": "window_lfu",
                    "slots_per_layer": 496,
                    "expert_accesses": 100,
                    "resident_hits": 98,
                    "blocking_misses": 2,
                    "miss_bytes": 200,
                    "decode_tokens": 2,
                    "network_seconds_lower_bound": 0.25,
                    "estimated_miss_ms_per_token": 7.5,
                    "estimated_slowdown_ratio": 1.075,
                }
            ],
        }
    ]

    result = summarize(rows, [496], ["window_lfu"])

    profile = result["simulations"]["window_lfu:496"]
    assert result["f1"] == 0.5
    assert profile["byte_hit_rate"] == 0.98
    assert profile["blocking_misses"] == 2
