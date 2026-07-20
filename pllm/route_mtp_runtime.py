from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .route_mtp import RouteMTPCheckpoint
from .route_mtp_torch import (
    RouteMTPAttentionState,
    TorchRouteMTPProbe,
    TorchTargetRouteHeads,
)


@dataclass(slots=True, frozen=True)
class RouteMTPPrediction:
    request_id: str
    token_id: int
    mtp_experts: tuple[int, ...]
    direct_scores: dict[int, Any]
    latency_ms: float
    route_hidden: Any | None = None


@dataclass(slots=True, frozen=True)
class RouteMTPPairedObservation:
    actual_by_layer: dict[int, list[int]]
    prediction: RouteMTPPrediction


class RouteMTPShadowPairer:
    """Order-independent one-token shift alignment for shadow observations.

    Prediction i is produced after actual route i and therefore evaluates
    against actual route i+1. Actual-route delivery can lag GPU sampling under
    vLLM async scheduling, so pairing is based on sequence indices rather than
    callback order.
    """

    def __init__(self, max_pending_steps: int = 64) -> None:
        if max_pending_steps <= 0:
            raise ValueError("max_pending_steps must be positive")
        self.max_pending_steps = int(max_pending_steps)
        self.predictions: dict[int, RouteMTPPrediction] = {}
        self.actuals: dict[int, dict[int, list[int]]] = {}
        self.prediction_steps = 0
        self.prediction_count = 0
        self.actual_steps = 0
        self.paired_predictions = 0
        self.dropped_predictions = 0
        self.dropped_actuals = 0

    def add_prediction(
        self,
        prediction: RouteMTPPrediction,
        *,
        target_actual_index: int | None = None,
    ) -> None:
        target = (
            self.prediction_steps + 1
            if target_actual_index is None
            else int(target_actual_index)
        )
        if target <= 0:
            raise ValueError("RouteMTP target actual index must be positive")
        self.predictions[target] = prediction
        self.prediction_steps = max(self.prediction_steps + 1, target)
        self.prediction_count += 1

    def add_actual(
        self, actual_by_layer: Mapping[int, Sequence[int]]
    ) -> list[RouteMTPPairedObservation]:
        actual_index = self.actual_steps
        self.actual_steps += 1
        if actual_index > 0:
            self.actuals[actual_index] = {
                int(layer): [int(expert) for expert in experts]
                for layer, experts in actual_by_layer.items()
            }
        return self._take_ready()

    def reset(self) -> None:
        self.predictions.clear()
        self.actuals.clear()
        self.prediction_steps = 0
        self.prediction_count = 0
        self.actual_steps = 0
        self.paired_predictions = 0
        self.dropped_predictions = 0
        self.dropped_actuals = 0

    def status(self) -> dict[str, int]:
        return {
            "predictions": self.prediction_count,
            "actual_steps": self.actual_steps,
            "paired_steps": self.paired_predictions,
            "pending_predictions": len(self.predictions),
            "pending_actual_steps": len(self.actuals),
            "dropped_predictions": self.dropped_predictions,
            "dropped_actual_steps": self.dropped_actuals,
        }

    def _take_ready(self) -> list[RouteMTPPairedObservation]:
        ready: list[RouteMTPPairedObservation] = []
        for actual_index in sorted(set(self.predictions) & set(self.actuals)):
            ready.append(
                RouteMTPPairedObservation(
                    actual_by_layer=self.actuals.pop(actual_index),
                    prediction=self.predictions.pop(actual_index),
                )
            )
            self.paired_predictions += 1
        while len(self.predictions) > self.max_pending_steps:
            self.predictions.pop(min(self.predictions))
            self.dropped_predictions += 1
        while len(self.actuals) > self.max_pending_steps:
            self.actuals.pop(min(self.actuals))
            self.dropped_actuals += 1
        return ready


class RouteMTPGPUShadowBridge:
    """Run the checkpoint MTP router probe after vLLM samples one token.

    The bridge is restricted to one request. It primes a single prompt chunk,
    then advances one decode token at a time. It emits calibration observations
    only and never changes the target router result or resident expert set.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str,
        *,
        capture_features: bool = False,
        load_target_heads: bool = True,
    ) -> None:
        checkpoint = RouteMTPCheckpoint.from_model(model_path)
        self.checkpoint = checkpoint
        self.device = device
        self.probe = TorchRouteMTPProbe.from_checkpoint(
            checkpoint,
            device=device,
            allow_accelerator=device != "cpu",
        )
        self.target_heads = (
            TorchTargetRouteHeads.from_checkpoint(
                checkpoint,
                device=device,
                allow_accelerator=device != "cpu",
            )
            if load_target_heads
            else None
        )
        self.capture_features = bool(capture_features)
        self.pairer = RouteMTPShadowPairer()
        self.attention_states: dict[str, RouteMTPAttentionState] = {}
        self.sample_calls = 0
        self.skipped_calls = 0
        self.failed_calls = 0
        self.last_skip_reason = ""
        self.last_latency_ms = 0.0
        self.total_latency_ms = 0.0
        self.decode_sample_steps = 0

    def observe_sample(
        self,
        model_runner: Any,
        scheduler_output: Any,
        sampled_token_ids: Any,
        sample_hidden_states: Any,
        full_hidden_states: Any | None = None,
        *,
        emit_prediction: bool = True,
    ) -> RouteMTPPrediction | None:
        import torch

        req_ids = tuple(getattr(model_runner.input_batch, "req_ids", ()))
        if len(req_ids) != 1:
            return self._skip("requires_exactly_one_request")
        request_id = str(req_ids[0])
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
        scheduled_tokens = int(scheduled.get(request_id, 0))
        if scheduled_tokens <= 0:
            return self._skip("requires_scheduled_tokens")
        if not isinstance(sampled_token_ids, torch.Tensor):
            return self._skip("sampled_tokens_not_on_accelerator")
        flattened = sampled_token_ids.reshape(sampled_token_ids.shape[0], -1)
        if flattened.shape[0] != 1 or flattened.shape[1] < 1:
            return self._skip("unexpected_sampled_token_shape")
        token = flattened[:, 0]
        if bool((token < 0).any()):
            return self._skip("invalid_sampled_token")
        if sample_hidden_states.ndim != 2 or sample_hidden_states.shape[0] != 1:
            return self._skip("unexpected_target_hidden_shape")
        if scheduled_tokens > 1 and (
            full_hidden_states is None
            or full_hidden_states.ndim != 2
            or full_hidden_states.shape[0] < scheduled_tokens
        ):
            return self._skip("prompt_hidden_states_unavailable")
        computed_tokens = getattr(
            model_runner.input_batch, "num_computed_tokens_cpu", None
        )
        previously_computed = (
            int(computed_tokens[0]) if computed_tokens is not None else 0
        )
        if request_id not in self.attention_states and previously_computed > 0:
            return self._skip("chunked_prompt_not_supported")

        target_actual_index = self.decode_sample_steps + 1
        self.decode_sample_steps += 1
        started = time.perf_counter_ns()
        state = self.attention_states.get(request_id, RouteMTPAttentionState())
        if scheduled_tokens == 1:
            shifted_ids = token
            target_rows = sample_hidden_states
        else:
            target_ids = model_runner.input_ids.gpu[:scheduled_tokens]
            shifted_ids = torch.cat((target_ids[1:], token.to(target_ids.dtype)))
            target_rows = full_hidden_states[:scheduled_tokens]
        embeddings = model_runner.get_model().embed_input_ids(shifted_ids)
        probe_output = None
        for row_index in range(scheduled_tokens):
            probe_output = self.probe.forward(
                embeddings[row_index : row_index + 1],
                target_rows[row_index : row_index + 1],
                state,
            )
            state = probe_output.state
        assert probe_output is not None
        direct_scores: dict[int, Any] = {}
        if self.target_heads is not None:
            target_output = self.target_heads.forward(probe_output.route_hidden)
            # This D2H copy is isolated to shadow calibration. The exact target
            # route remains on vLLM's own path.
            score_rows = (
                target_output.routing_scores[0]
                .to(dtype=torch.float16)
                .cpu()
                .numpy()
            )
            direct_scores = {
                layer: score_rows[index]
                for index, layer in enumerate(target_output.layers)
            }
        route_hidden = None
        if self.capture_features:
            route_hidden = (
                probe_output.route_hidden[0]
                .to(dtype=torch.float16)
                .cpu()
                .numpy()
                .copy()
            )
        mtp_experts = tuple(
            int(value) for value in probe_output.topk_experts[0].tolist()
        )
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        prediction = RouteMTPPrediction(
            request_id=request_id,
            token_id=int(token.item()),
            mtp_experts=mtp_experts,
            direct_scores=direct_scores,
            latency_ms=latency_ms,
            route_hidden=route_hidden,
        )
        self.attention_states[request_id] = probe_output.state
        if emit_prediction:
            self.pairer.add_prediction(
                prediction,
                target_actual_index=target_actual_index,
            )
        self.sample_calls += 1
        self.last_latency_ms = latency_ms
        self.total_latency_ms += latency_ms
        return prediction

    def observe_actual(
        self, actual_by_layer: Mapping[int, Sequence[int]]
    ) -> list[RouteMTPPairedObservation]:
        return self.pairer.add_actual(actual_by_layer)

    def reset_request(self) -> None:
        self.attention_states.clear()
        self.pairer.reset()
        self.decode_sample_steps = 0

    def status(self) -> dict[str, Any]:
        allocated = self.probe.allocated_bytes() + (
            self.target_heads.allocated_bytes() if self.target_heads is not None else 0
        )
        return {
            "enabled": True,
            "mode": "gpu_shadow_only",
            "device": self.device,
            "allocated_bytes": allocated,
            "sample_calls": self.sample_calls,
            "skipped_calls": self.skipped_calls,
            "failed_calls": self.failed_calls,
            "last_skip_reason": self.last_skip_reason,
            "last_latency_ms": self.last_latency_ms,
            "mean_latency_ms": (
                self.total_latency_ms / self.sample_calls if self.sample_calls else 0.0
            ),
            "target_routes_calibrated": False,
            "target_heads_loaded": self.target_heads is not None,
            "feature_capture_enabled": self.capture_features,
            "eviction_enabled": False,
            **self.pairer.status(),
        }

    def record_failure(self, reason: str) -> None:
        self.failed_calls += 1
        self.last_skip_reason = reason

    def _skip(self, reason: str) -> None:
        self.skipped_calls += 1
        self.last_skip_reason = reason
        return None
