from __future__ import annotations

import base64
import io

import numpy as np

from scripts.benchmark_decode_routes import decode_routes, summarize


def test_decode_routes_rejects_non_tensor_layout() -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((4, 3, 2), dtype=np.uint16), allow_pickle=False)

    decoded = decode_routes(base64.b64encode(buffer.getvalue()).decode())

    assert decoded.shape == (4, 3, 2)
    assert decoded.dtype == np.uint16


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
