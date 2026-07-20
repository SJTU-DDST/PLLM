from __future__ import annotations

import numpy as np
import torch

from pllm.route_adapter import SharedLowRankRouteAdapter, load_route_trace
from pllm.route_mtp_runtime import RouteMTPPrediction
from pllm.route_mtp_trace import RouteMTPTraceWriter


def test_route_trace_writer_round_trip(tmp_path) -> None:
    writer = RouteMTPTraceWriter(
        tmp_path, layers=(1, 3), hidden_size=4, active_experts=2
    )
    writer.append(
        RouteMTPPrediction(
            request_id="request-a",
            token_id=7,
            mtp_experts=(2, 5),
            direct_scores={},
            latency_ms=1.0,
            route_hidden=np.asarray([1, 2, 3, 4], dtype=np.float16),
        ),
        {1: [0, 2], 3: [4, 6]},
    )

    result = writer.flush()
    trace = load_route_trace(result["files"][0])

    assert result["samples"] == 1
    assert trace.request_id == "request-a"
    assert trace.layers == (1, 3)
    assert trace.features.shape == (1, 4)
    assert trace.actual_routes.tolist() == [[[0, 2], [4, 6]]]
    assert trace.mtp_routes.tolist() == [[2, 5]]


def test_shared_route_adapter_has_layer_specific_outputs(tmp_path) -> None:
    model = SharedLowRankRouteAdapter(4, 2, 8, 3)
    output = model(torch.ones(5, 4, dtype=torch.float16))

    assert output.shape == (5, 2, 8)
    assert model.parameter_count() == 4 * 3 + 2 * 8 * 3 + 2 * 8

    path = tmp_path / "adapter.pt"
    model.save(path, metadata={"experiment": "test"})
    restored, metadata = SharedLowRankRouteAdapter.load(path)
    assert restored.config() == model.config()
    assert metadata == {"experiment": "test"}
