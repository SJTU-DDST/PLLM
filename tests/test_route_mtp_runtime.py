from __future__ import annotations

from types import SimpleNamespace

import torch

from pllm.route_mtp_runtime import (
    RouteMTPGPUShadowBridge,
    RouteMTPPrediction,
    RouteMTPShadowPairer,
)
from tests.test_route_mtp import _write_fake_mtp


def _prediction(token_id: int = 7) -> RouteMTPPrediction:
    return RouteMTPPrediction(
        request_id="request-1",
        token_id=token_id,
        mtp_experts=(1, 2),
        direct_scores={0: (0.1, 0.2)},
        latency_ms=1.0,
    )


def test_shadow_pairer_aligns_prediction_with_following_actual_route() -> None:
    pairer = RouteMTPShadowPairer()
    pairer.add_actual({0: [0, 1]})
    pairer.add_prediction(_prediction())

    ready = pairer.add_actual({0: [2, 3]})

    assert len(ready) == 1
    assert ready[0].actual_by_layer == {0: [2, 3]}
    assert ready[0].prediction.token_id == 7
    assert pairer.status()["paired_steps"] == 1


def test_shadow_pairer_is_independent_of_async_callback_order() -> None:
    pairer = RouteMTPShadowPairer()
    pairer.add_prediction(_prediction())

    assert pairer.add_actual({0: [0, 1]}) == []
    ready = pairer.add_actual({0: [2, 3]})

    assert len(ready) == 1
    assert ready[0].actual_by_layer == {0: [2, 3]}


def test_shadow_pairer_reports_sparse_emissions_as_prediction_count() -> None:
    pairer = RouteMTPShadowPairer()
    pairer.add_prediction(_prediction(), target_actual_index=4)

    assert pairer.status()["predictions"] == 1
    assert pairer.status()["pending_predictions"] == 1


def test_gpu_shadow_bridge_runs_tiny_checkpoint_on_cpu(tmp_path) -> None:
    model_path = _write_fake_mtp(tmp_path / "model")
    bridge = RouteMTPGPUShadowBridge(model_path, "cpu")
    embedding = torch.nn.Embedding(16, 4, dtype=torch.bfloat16)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["request-1"]),
        input_ids=SimpleNamespace(gpu=torch.tensor([1, 2, 3], dtype=torch.int64)),
        get_model=lambda: SimpleNamespace(embed_input_ids=embedding),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-1": 1}
    )

    bridge.observe_actual({0: [0, 1], 2: [2, 3]})
    prediction = bridge.observe_sample(
        runner,
        scheduler_output,
        torch.tensor([[5]], dtype=torch.int64),
        torch.ones(1, 4, dtype=torch.bfloat16),
    )
    paired = bridge.observe_actual({0: [4, 5], 2: [6, 7]})

    assert prediction is not None
    assert prediction.token_id == 5
    assert len(prediction.mtp_experts) == 2
    assert set(prediction.direct_scores) == {0, 2}
    assert all(len(scores) == 8 for scores in prediction.direct_scores.values())
    assert len(paired) == 1
    assert paired[0].prediction is prediction
    assert bridge.status()["eviction_enabled"] is False


def test_gpu_shadow_bridge_primes_prompt_with_shifted_tokens(tmp_path) -> None:
    bridge = RouteMTPGPUShadowBridge(
        _write_fake_mtp(tmp_path / "model"), "cpu"
    )
    embedding = torch.nn.Embedding(16, 4, dtype=torch.bfloat16)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["request-1"]),
        input_ids=SimpleNamespace(gpu=torch.tensor([3, 4, 5], dtype=torch.int64)),
        get_model=lambda: SimpleNamespace(embed_input_ids=embedding),
    )

    prediction = bridge.observe_sample(
        runner,
        SimpleNamespace(num_scheduled_tokens={"request-1": 3}),
        torch.tensor([[6]], dtype=torch.int64),
        torch.ones(1, 4, dtype=torch.bfloat16),
        torch.arange(12, dtype=torch.bfloat16).view(3, 4),
    )

    assert prediction is not None
    assert prediction.token_id == 6
    assert bridge.attention_states["request-1"].tokens == 3
    assert bridge.status()["skipped_calls"] == 0


def test_gpu_shadow_bridge_skips_prefill_and_batches(tmp_path) -> None:
    bridge = RouteMTPGPUShadowBridge(
        _write_fake_mtp(tmp_path / "model"), "cpu"
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )

    result = bridge.observe_sample(
        runner,
        SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1}),
        torch.tensor([[1], [2]]),
        torch.ones(2, 4, dtype=torch.bfloat16),
    )

    assert result is None
    assert bridge.status()["last_skip_reason"] == "requires_exactly_one_request"


def test_gpu_shadow_bridge_rejects_unprimed_chunked_prompt(tmp_path) -> None:
    bridge = RouteMTPGPUShadowBridge(
        _write_fake_mtp(tmp_path / "model"), "cpu"
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["request-1"],
            num_computed_tokens_cpu=[512],
        ),
    )

    result = bridge.observe_sample(
        runner,
        SimpleNamespace(num_scheduled_tokens={"request-1": 1}),
        torch.tensor([[1]]),
        torch.ones(1, 4, dtype=torch.bfloat16),
    )

    assert result is None
    assert bridge.status()["last_skip_reason"] == "chunked_prompt_not_supported"
