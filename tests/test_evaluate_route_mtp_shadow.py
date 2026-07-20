from __future__ import annotations

from pathlib import Path

import numpy as np

from pllm.expert_catalog import ExpertCatalog
from scripts.evaluate_route_mtp_shadow import evaluate_route_files


def _catalog() -> ExpertCatalog:
    return ExpertCatalog(
        model_path="fake",
        architecture="TinyMoE",
        num_hidden_layers=2,
        moe_layers=[0, 1],
        experts_per_layer=4,
        active_experts_per_token=1,
        total_tensor_bytes=0,
        routed_expert_bytes=0,
        non_routed_bytes=0,
        experts=[],
    )


def test_cpu_shadow_replay_marks_missing_mtp_signal_honestly(tmp_path: Path) -> None:
    path = tmp_path / "routes.npz"
    decode = np.zeros((5, 2, 1), dtype=np.uint16)
    np.savez_compressed(path, decode=decode)

    result = evaluate_route_files(
        [path],
        _catalog(),
        [3, 4],
        target_miss_rate=0.5,
        confidence_delta=0.9,
        minimum_samples=1,
    )

    assert result["gpu_used"] is False
    assert result["decode_tokens"] == 5
    assert result["mtp_signal_rows"] == 0
    assert result["evidence"].endswith("no_mtp_signal")


def test_cpu_shadow_replay_consumes_explicit_mtp_routes(tmp_path: Path) -> None:
    path = tmp_path / "routes.npz"
    decode = np.zeros((3, 2, 1), dtype=np.uint16)
    mtp_routes = np.ones((3, 1), dtype=np.uint16)
    np.savez_compressed(path, decode=decode, mtp_routes=mtp_routes)

    result = evaluate_route_files([path], _catalog(), [3, 4])

    assert result["mtp_signal_rows"] == 3
    assert result["predictor"]["mtp_signal_attached"] is True
