from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .route_mtp import (
    ROUTER_PROBE_PROFILE,
    TARGET_GATE_PATTERN,
    RouteMTPCheckpoint,
    RouteMTPTensorLoader,
)


@dataclass(slots=True)
class RouteMTPAttentionState:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    @property
    def tokens(self) -> int:
        return int(self.keys.shape[2]) if self.keys is not None else 0


@dataclass(slots=True)
class RouteMTPProbeOutput:
    topk_experts: torch.Tensor
    router_logits: torch.Tensor
    route_hidden: torch.Tensor
    state: RouteMTPAttentionState


@dataclass(slots=True)
class TargetRouteHeadOutput:
    layers: tuple[int, ...]
    topk_experts: torch.Tensor
    router_logits: torch.Tensor
    routing_scores: torch.Tensor


class TorchTargetRouteHeads:
    """Original target gates evaluated on MTP hidden as an uncalibrated prior."""

    def __init__(
        self,
        checkpoint: RouteMTPCheckpoint,
        layers: tuple[int, ...],
        weights: torch.Tensor,
        correction_bias: torch.Tensor,
    ) -> None:
        self.checkpoint = checkpoint
        self.layers = layers
        self.weights = weights
        self.correction_bias = correction_bias

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: RouteMTPCheckpoint,
        *,
        device: str = "cpu",
        allow_accelerator: bool = False,
    ) -> "TorchTargetRouteHeads":
        layers = checkpoint.target_route_layers
        layer_to_index = {layer: index for index, layer in enumerate(layers)}
        weights = torch.empty(
            len(layers),
            checkpoint.expert_count,
            checkpoint.hidden_size,
            # vLLM's Nemotron gate forces FP32 compute. Keep one FP32 route-head
            # image so the shadow path does not recast 40 matrices per token.
            dtype=torch.float32,
            device=device,
        )
        correction_bias = torch.empty(
            len(layers),
            checkpoint.expert_count,
            dtype=torch.float32,
            device=device,
        )
        loaded = set()
        loader = RouteMTPTensorLoader(checkpoint)
        for name, tensor in loader.iter_target_gate_tensors(
            device=device,
            allow_accelerator=allow_accelerator,
        ):
            match = TARGET_GATE_PATTERN.match(name)
            if match is None:
                raise ValueError(f"unexpected target gate tensor: {name}")
            index = layer_to_index[int(match.group("layer"))]
            if name.endswith(".gate.weight"):
                weights[index].copy_(tensor)
            else:
                correction_bias[index].copy_(tensor)
            loaded.add(name)
        if loaded != {item.name for item in checkpoint.target_gate_tensors}:
            raise RuntimeError("target route head loader returned an incomplete set")
        return cls(checkpoint, layers, weights, correction_bias)

    @torch.inference_mode()
    def forward(self, future_hidden: torch.Tensor) -> TargetRouteHeadOutput:
        if future_hidden.ndim != 2 or future_hidden.shape[1] != self.checkpoint.hidden_size:
            raise ValueError(
                f"future hidden must have shape [batch, {self.checkpoint.hidden_size}]"
            )
        hidden = future_hidden.to(device=self.weights.device, dtype=torch.float32)
        router_logits = torch.einsum("bh,leh->ble", hidden, self.weights)
        routing_scores = torch.sigmoid(router_logits) + self.correction_bias.unsqueeze(0)
        topk = routing_scores.topk(
            self.checkpoint.active_experts, dim=-1
        ).indices
        return TargetRouteHeadOutput(
            self.layers,
            topk,
            router_logits,
            routing_scores,
        )

    def allocated_bytes(self) -> int:
        return (
            self.weights.untyped_storage().nbytes()
            + self.correction_bias.untyped_storage().nbytes()
        )


class TorchRouteMTPProbe:
    """Minimal exact-weight Nemotron MTP forward ending at its own MoE gate.

    This probe deliberately does not execute the MTP experts and does not claim
    to predict the target model's 40 layer-local routes.
    """

    def __init__(
        self,
        checkpoint: RouteMTPCheckpoint,
        weights: dict[str, torch.Tensor],
    ) -> None:
        self.checkpoint = checkpoint
        self.weights = weights
        self.device = next(iter(weights.values())).device
        self.dtype = next(
            tensor.dtype for tensor in weights.values() if tensor.dtype != torch.float32
        )
        required = {
            "mtp.layers.0.enorm.weight",
            "mtp.layers.0.hnorm.weight",
            "mtp.layers.0.eh_proj.weight",
            "mtp.layers.0.norm.weight",
            "mtp.layers.0.mixer.q_proj.weight",
            "mtp.layers.0.mixer.k_proj.weight",
            "mtp.layers.0.mixer.v_proj.weight",
            "mtp.layers.0.mixer.o_proj.weight",
            "mtp.layers.1.norm.weight",
            "mtp.layers.1.mixer.gate.weight",
            "mtp.layers.1.mixer.gate.e_score_correction_bias",
        }
        missing = sorted(required - weights.keys())
        if missing:
            raise ValueError(f"RouteMTP probe weights are incomplete: {missing}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: RouteMTPCheckpoint,
        *,
        device: str = "cpu",
        allow_accelerator: bool = False,
    ) -> "TorchRouteMTPProbe":
        weights = dict(
            RouteMTPTensorLoader(checkpoint).iter_tensors(
                ROUTER_PROBE_PROFILE,
                device=device,
                allow_accelerator=allow_accelerator,
            )
        )
        return cls(checkpoint, weights)

    @torch.inference_mode()
    def forward(
        self,
        token_embeddings: torch.Tensor,
        target_hidden: torch.Tensor,
        state: RouteMTPAttentionState | None = None,
    ) -> RouteMTPProbeOutput:
        if token_embeddings.shape != target_hidden.shape:
            raise ValueError("token embeddings and target hidden states must match")
        if token_embeddings.ndim != 2 or token_embeddings.shape[1] != self.checkpoint.hidden_size:
            raise ValueError(
                f"RouteMTP inputs must have shape [batch, {self.checkpoint.hidden_size}]"
            )
        token_embeddings = token_embeddings.to(device=self.device, dtype=self.dtype)
        target_hidden = target_hidden.to(device=self.device, dtype=self.dtype)
        state = state or RouteMTPAttentionState()

        embeddings_normed = self._rms_norm(
            token_embeddings, self.weights["mtp.layers.0.enorm.weight"]
        )
        hidden_normed = self._rms_norm(
            target_hidden, self.weights["mtp.layers.0.hnorm.weight"]
        )
        fused = F.linear(
            torch.cat((embeddings_normed, hidden_normed), dim=-1),
            self.weights["mtp.layers.0.eh_proj.weight"],
        )
        attention_input = self._rms_norm(
            fused, self.weights["mtp.layers.0.norm.weight"]
        )
        attention_output, state = self._attention(attention_input, state)

        # vLLM RMSNorm(hidden_states, residual) adds the attention output to the
        # fusion residual before normalizing for the following MoE layer.
        route_residual = fused + attention_output
        route_hidden = self._rms_norm(
            route_residual, self.weights["mtp.layers.1.norm.weight"]
        )
        router_logits = F.linear(
            route_hidden.float(),
            self.weights["mtp.layers.1.mixer.gate.weight"].float(),
        )
        topk = self._topk(router_logits)
        return RouteMTPProbeOutput(topk, router_logits, route_hidden, state)

    def allocated_bytes(self) -> int:
        return sum(
            tensor.untyped_storage().nbytes()
            for tensor in self.weights.values()
        )

    def _attention(
        self,
        hidden: torch.Tensor,
        state: RouteMTPAttentionState,
    ) -> tuple[torch.Tensor, RouteMTPAttentionState]:
        batch = hidden.shape[0]
        heads = self.checkpoint.num_attention_heads
        kv_heads = self.checkpoint.num_key_value_heads
        head_dim = self.checkpoint.head_dim
        query = F.linear(
            hidden, self.weights["mtp.layers.0.mixer.q_proj.weight"]
        ).view(batch, heads, head_dim)
        key = F.linear(
            hidden, self.weights["mtp.layers.0.mixer.k_proj.weight"]
        ).view(batch, kv_heads, head_dim)
        value = F.linear(
            hidden, self.weights["mtp.layers.0.mixer.v_proj.weight"]
        ).view(batch, kv_heads, head_dim)
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
        if state.keys is not None:
            if state.keys.shape[:2] != key.shape[:2]:
                raise ValueError("RouteMTP attention cache batch/head shape changed")
            key = torch.cat((state.keys, key), dim=2)
            value = torch.cat((state.values, value), dim=2)
        state = RouteMTPAttentionState(key, value)

        repeats = heads // kv_heads
        expanded_key = key.repeat_interleave(repeats, dim=1)
        expanded_value = value.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("bhd,bhtd->bht", query, expanded_key)
        scores = scores * (head_dim**-0.5)
        probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        context = torch.einsum("bht,bhtd->bhd", probabilities, expanded_value)
        context = context.reshape(batch, heads * head_dim)
        output = F.linear(
            context, self.weights["mtp.layers.0.mixer.o_proj.weight"]
        )
        return output, state

    def _topk(self, router_logits: torch.Tensor) -> torch.Tensor:
        choice = torch.sigmoid(router_logits)
        choice = choice + self.weights[
            "mtp.layers.1.mixer.gate.e_score_correction_bias"
        ].float()
        groups = self.checkpoint.expert_groups
        if groups > 1:
            grouped = choice.view(
                -1, groups, self.checkpoint.expert_count // groups
            )
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            selected_groups = group_scores.topk(
                self.checkpoint.topk_groups, dim=-1
            ).indices
            mask = torch.zeros_like(group_scores, dtype=torch.bool)
            mask.scatter_(1, selected_groups, True)
            choice = choice.masked_fill(
                ~mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice),
                float("-inf"),
            )
        return choice.topk(self.checkpoint.active_experts, dim=-1).indices

    def _rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value.float() * torch.rsqrt(
            variance + self.checkpoint.norm_epsilon
        )
        return (normalized * weight.float()).to(value.dtype)
