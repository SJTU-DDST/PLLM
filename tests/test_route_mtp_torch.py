from __future__ import annotations

import torch

from pllm.route_mtp import RouteMTPCheckpoint
from pllm.route_mtp_torch import (
    RouteMTPAttentionState,
    TorchRouteMTPProbe,
    TorchTargetRouteHeads,
)
from tests.test_route_mtp import _write_fake_mtp


def test_torch_route_mtp_probe_runs_to_gate_without_expert_weights(tmp_path) -> None:
    checkpoint = RouteMTPCheckpoint.from_model(_write_fake_mtp(tmp_path / "model"))
    probe = TorchRouteMTPProbe.from_checkpoint(checkpoint)
    embedding = torch.ones(1, 4, dtype=torch.bfloat16)
    hidden = torch.arange(4, dtype=torch.bfloat16).view(1, 4)

    first = probe.forward(embedding, hidden, RouteMTPAttentionState())
    second = probe.forward(embedding, hidden, first.state)

    assert first.topk_experts.shape == (1, 2)
    assert first.topk_experts.unique().numel() == 2
    assert first.router_logits.shape == (1, 8)
    assert torch.isfinite(first.router_logits).all()
    assert first.state.tokens == 1
    assert second.state.tokens == 2
    assert not any(".mixer.experts." in name for name in probe.weights)


def test_original_target_route_heads_project_mtp_hidden_for_every_layer(tmp_path) -> None:
    checkpoint = RouteMTPCheckpoint.from_model(_write_fake_mtp(tmp_path / "model"))
    heads = TorchTargetRouteHeads.from_checkpoint(checkpoint)

    output = heads.forward(torch.ones(1, 4, dtype=torch.bfloat16))

    assert output.layers == (0, 2)
    assert output.router_logits.shape == (1, 2, 8)
    assert output.topk_experts.shape == (1, 2, 2)
    assert output.topk_experts[0, 0].unique().numel() == 2
    assert output.routing_scores.shape == output.router_logits.shape
    assert torch.isfinite(output.routing_scores).all()
