from __future__ import annotations

import copy
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Any

from .config import PLLMConfig
from .expert_catalog import GIB, ExpertCatalog
from .expert_residency import ResidencyPlanner, ResourceEnvelope
from .decode_residency import HorizonAwareLayerPlanner
from .models import SensorSnapshot, WorkloadClass


WORKLOAD_ENVELOPES: dict[WorkloadClass, dict[str, float]] = {
    WorkloadClass.IDLE: {
        "foreground_reserve_gib": 20.0,
        "compute_duty_cycle": 1.0,
    },
    WorkloadClass.CREATIVE: {
        "foreground_reserve_gib": 64.0,
        "compute_duty_cycle": 0.35,
    },
    WorkloadClass.GAME: {
        "foreground_reserve_gib": 72.0,
        "compute_duty_cycle": 0.15,
    },
    WorkloadClass.GPU_PRESSURE: {
        "foreground_reserve_gib": 48.0,
        "compute_duty_cycle": 0.3,
    },
    WorkloadClass.MEMORY_PRESSURE: {
        "foreground_reserve_gib": 104.0,
        "compute_duty_cycle": 0.1,
    },
    WorkloadClass.POWER_PRESSURE: {
        "foreground_reserve_gib": 64.0,
        "compute_duty_cycle": 0.2,
    },
}


DISCRETE_GPU_ENVELOPES: dict[WorkloadClass, dict[str, float]] = {
    WorkloadClass.IDLE: {"foreground_reserve_gib": 4.0, "compute_duty_cycle": 1.0},
    WorkloadClass.CREATIVE: {
        "foreground_reserve_gib": 10.0,
        "compute_duty_cycle": 0.35,
    },
    WorkloadClass.GAME: {"foreground_reserve_gib": 16.0, "compute_duty_cycle": 0.15},
    WorkloadClass.GPU_PRESSURE: {
        "foreground_reserve_gib": 8.0,
        "compute_duty_cycle": 0.3,
    },
    WorkloadClass.MEMORY_PRESSURE: {
        "foreground_reserve_gib": 40.0,
        "compute_duty_cycle": 0.1,
    },
    WorkloadClass.POWER_PRESSURE: {
        "foreground_reserve_gib": 16.0,
        "compute_duty_cycle": 0.2,
    },
}


class ExpertResidencyControlPlane:
    """Read-only expert residency recommendations until a vLLM slot backend exists."""

    def __init__(self, config: PLLMConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._catalog: ExpertCatalog | None = None
        self._planner: ResidencyPlanner | None = None
        self._error = ""
        self._last_plan: dict[str, Any] = {}
        self._last_updated_at = 0.0
        self._capacity_generation = 0
        self._capacity_key: tuple[Any, ...] | None = None
        self._last_decode_plan: dict[str, Any] = {}
        self._last_decode_key: tuple[Any, ...] | None = None
        self._decode_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pllm-residency"
        )
        self._decode_future: concurrent.futures.Future[dict[str, Any]] | None = None
        self._decode_future_key: tuple[Any, ...] | None = None
        self._decode_async_result: dict[str, Any] = {}
        self._decode_async_result_key: tuple[Any, ...] | None = None
        self._load_catalog()

    def _load_catalog(self) -> None:
        try:
            catalog = ExpertCatalog.from_model(Path(self.config.model_path))
            self._catalog = catalog
            self._planner = ResidencyPlanner(
                catalog,
                slot_profiles=tuple(
                    sorted(
                        set(self.config.decode_candidate_slots)
                        | {catalog.experts_per_layer}
                    )
                ),
            )
        except (OSError, ValueError, TypeError) as exc:
            self._error = str(exc)

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._catalog is None:
                return {
                    "available": False,
                    "enabled": self.config.expert_residency_enabled,
                    "data_plane_ready": False,
                    "backend": "unavailable",
                    "error": self._error,
                    "evidence": "catalog_unavailable",
                    "plan": {},
                }
            catalog = self._catalog
            return {
                "available": True,
                "enabled": self.config.expert_residency_enabled,
                "data_plane_ready": False,
                "backend": "control_plane_only",
                "error": self._error,
                "evidence": "checkpoint_header_projection",
                "model": {
                    "architecture": catalog.architecture,
                    "moe_layers": len(catalog.moe_layers),
                    "experts_per_layer": catalog.experts_per_layer,
                    "top_k": catalog.active_experts_per_token,
                    "expert_object_count": len(catalog.experts),
                    "routed_expert_gib": round(
                        catalog.routed_expert_bytes / 1024**3, 3
                    ),
                    "non_routed_gib": round(catalog.non_routed_bytes / 1024**3, 3),
                    "average_expert_mib": round(
                        catalog.average_expert_bytes / 1024**2, 3
                    ),
                },
                "plan": dict(self._last_plan),
                "decode_plan": dict(self._last_decode_plan),
                "updated_at": self._last_updated_at,
                "guardrail": "recommendation_only_no_vllm_weight_mutation",
            }

    def plan_decode_residency(self, runtime: dict[str, Any]) -> dict[str, Any]:
        if not self.config.decode_planner_async:
            return self._plan_decode_residency_sync(runtime)
        request_key = self._async_request_key(runtime)
        with self._lock:
            if (
                self._decode_async_result_key == request_key
                and self._decode_async_result
            ):
                return dict(self._decode_async_result)
            future = self._decode_future
            if future is not None and future.done():
                completed_key = self._decode_future_key
                self._decode_future = None
                self._decode_future_key = None
                try:
                    completed = future.result()
                except Exception as exc:
                    completed = {
                        "action": "yield",
                        "reason": f"background residency planner failed: {exc}",
                        "planner_error": True,
                    }
                if completed_key == request_key:
                    self._decode_async_result = dict(completed)
                    self._decode_async_result_key = completed_key
                    return dict(completed)
                future = None
            if future is None:
                self._decode_future_key = request_key
                self._decode_future = self._decode_executor.submit(
                    self._plan_decode_residency_sync, copy.deepcopy(runtime)
                )
        return self._pending_decode_plan(runtime)

    def _plan_decode_residency_sync(self, runtime: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._catalog is None:
                return {}
            catalog = self._catalog
            capacity_plan = dict(self._last_plan)
            cached_key = self._last_decode_key
            cached_plan = dict(self._last_decode_plan)

        trace = dict(runtime.get("route_trace") or {})
        phase = str(trace.get("phase", "idle"))
        observations = int(trace.get("decode_observations", 0))
        prediction = dict(trace.get("next_window") or {})
        layer_bytes: dict[int, int] = {
            layer: sum(
                item.size_bytes for item in catalog.experts if item.layer == layer
            )
            for layer in catalog.moe_layers
        }
        data_layers = dict(runtime.get("data_plane") or {}).get("layers") or []
        current_slots = {
            int(item["layer"]): int(item["slot_count"])
            for item in data_layers
            if "layer" in item and "slot_count" in item
        }
        capacity_action = str(capacity_plan.get("action", ""))
        if capacity_action == "full_resident":
            target_reclaim_bytes = 0
        elif capacity_action in {"elastic_resident", "hibernate", "yield"}:
            target_reclaim_bytes = int(
                float(capacity_plan.get("projected_reclaim_gib", 0.0)) * GIB
            )
        else:
            target_reclaim_bytes = int(self.config.decode_target_reclaim_gib * GIB)
        horizon = dict(runtime.get("decode_horizon") or {})
        remaining_tokens = int(horizon.get("remaining_tokens", 0))
        decode_requests = int(
            horizon.get("decode_requests", 1 if remaining_tokens > 0 else 0)
        )
        if decode_requests != 1:
            remaining_tokens = 0
        horizon_bucket = max(1, int(self.config.decode_horizon_bucket_tokens))
        planner_horizon = remaining_tokens // horizon_bucket * horizon_bucket
        decode_key = (
            phase,
            int(prediction.get("request_generation", -1)),
            int(prediction.get("route_generation", -1)),
            planner_horizon,
            target_reclaim_bytes // (16 * 1024**2),
            tuple(sorted(current_slots.items())),
            capacity_action,
            int(capacity_plan.get("generation", 0)),
        )
        if cached_key == decode_key and cached_plan:
            cached_plan["observations"] = observations
            cached_plan["horizon"] = horizon
            return cached_plan
        curve = dict(
            zip(
                self.config.decode_miss_batch_sizes,
                self.config.decode_miss_batch_p95_ms,
            )
        )
        if len(curve) != len(self.config.decode_miss_batch_sizes):
            raise ValueError("decode miss batch calibration arrays differ in length")
        planner = HorizonAwareLayerPlanner(
            top_k=catalog.active_experts_per_token,
            minimum_byte_hit_rate=self.config.decode_min_byte_hit_rate,
            maximum_slowdown_ratio=self.config.decode_max_slowdown_ratio,
            minimum_heldout_windows=self.config.decode_min_heldout_windows,
            miss_latency_curve_ms=curve,
        )
        result = planner.choose(
            phase=phase,
            prediction=prediction,
            candidate_slots=self.config.decode_candidate_slots,
            layer_bytes=layer_bytes,
            current_slots_by_layer=current_slots,
            target_reclaim_bytes=target_reclaim_bytes,
            io_budget_gib_s=self.config.expert_io_budget_gib_s,
            token_rate=self.config.expert_requested_token_rate,
            baseline_tpot_ms=self.config.decode_baseline_tpot_ms,
            remaining_decode_tokens=planner_horizon,
            resize_copy_gib_s=self.config.decode_resize_copy_gib_s,
            expand_gib_s=self.config.decode_expand_gib_s,
            rebuild_ms_per_layer=self.config.decode_rebuild_ms_per_layer,
            release_deadline_ms=self.config.expert_release_deadline_ms,
            full_slots=catalog.experts_per_layer,
        )
        decision = result.to_dict()
        if decision["action"] == "yield" and decode_requests != 1:
            decision["reason"] = (
                "elastic route learning requires exactly one active decode request"
            )
        if decision["action"] == "yield" and capacity_action == "hibernate":
            decision["action"] = "hibernate"
            decision["reason"] += "; capacity envelope requires deep release"
        resident_routed = sum(
            layer_bytes[layer]
            * int(result.slots_by_layer.get(layer, catalog.experts_per_layer))
            / catalog.experts_per_layer
            for layer in catalog.moe_layers
        )
        resident_weight = int(round(catalog.non_routed_bytes + resident_routed))
        decision.update(
            {
                "routed_bytes": int(round(resident_routed)),
                "resident_weight_bytes": resident_weight,
                "resident_weight_gib": round(resident_weight / GIB, 3),
                "projected_reclaim_gib": round(
                    (catalog.total_tensor_bytes - resident_weight) / GIB, 3
                ),
            }
        )
        decision["observations"] = observations
        decision["latency_guardrail"] = "heldout_next_window_strictly_below_5x"
        decision["horizon"] = horizon
        decision["horizon"]["planner_lower_bound_tokens"] = planner_horizon
        decision["request_generation"] = int(
            prediction.get("request_generation", -1)
        )
        decision["route_generation"] = int(prediction.get("route_generation", -1))
        decision["capacity_generation"] = int(capacity_plan.get("generation", 0))
        decision["target_reclaim_bytes"] = target_reclaim_bytes
        with self._lock:
            self._last_decode_plan = decision
            self._last_decode_key = decode_key
        return dict(decision)

    def _async_request_key(self, runtime: dict[str, Any]) -> tuple[Any, ...]:
        trace = dict(runtime.get("route_trace") or {})
        prediction = dict(trace.get("next_window") or {})
        horizon = dict(runtime.get("decode_horizon") or {})
        layers = dict(runtime.get("data_plane") or {}).get("layers") or []
        with self._lock:
            capacity_action = str(self._last_plan.get("action", ""))
            target_reclaim = int(
                float(self._last_plan.get("projected_reclaim_gib", 0.0))
                * GIB
            )
            capacity_generation = int(self._last_plan.get("generation", 0))
        return (
            str(trace.get("phase", "idle")),
            int(prediction.get("request_generation", -1)),
            int(prediction.get("route_generation", -1)),
            int(horizon.get("remaining_tokens", 0))
            // max(1, int(self.config.decode_horizon_bucket_tokens)),
            int(horizon.get("decode_requests", 0)),
            tuple(
                sorted(
                    (int(item.get("layer", -1)), int(item.get("slot_count", 0)))
                    for item in layers
                )
            ),
            capacity_action,
            target_reclaim // (16 * 1024**2),
            capacity_generation,
        )

    def _pending_decode_plan(self, runtime: dict[str, Any]) -> dict[str, Any]:
        horizon = dict(runtime.get("decode_horizon") or {})
        full_slots = self._catalog.experts_per_layer if self._catalog is not None else 0
        return {
            "action": "yield",
            "slots_per_layer": full_slots,
            "slots_by_layer": {
                str(layer): full_slots
                for layer in (self._catalog.moe_layers if self._catalog else ())
            },
            "projected_byte_hit_rate": 0.0,
            "estimated_slowdown_ratio": float("inf"),
            "reason": "background Pareto planner pending; use fast token-boundary yield",
            "planner_pending": True,
            "horizon": horizon,
            "evidence": "asynchronous_planner_pending_not_a_residency_decision",
        }

    def recommend(
        self, snapshot: SensorSnapshot, workload: WorkloadClass
    ) -> dict[str, Any]:
        if not snapshot.uma and snapshot.gpu_memory_total_gb:
            values = DISCRETE_GPU_ENVELOPES[workload]
            total_memory = snapshot.gpu_memory_total_gb
            capacity_scope = "discrete_gpu_vram"
        else:
            values = WORKLOAD_ENVELOPES[workload]
            total_memory = snapshot.memory_total_gb or 128.0
            capacity_scope = "coherent_uma"
        envelope = ResourceEnvelope(
            total_memory_gib=total_memory,
            foreground_reserve_gib=values["foreground_reserve_gib"],
            system_reserve_gib=self.config.expert_system_reserve_gib,
            io_budget_gib_s=self.config.expert_io_budget_gib_s,
            compute_duty_cycle=values["compute_duty_cycle"],
            requested_token_rate=self.config.expert_requested_token_rate,
            minimum_token_rate=self.config.expert_minimum_token_rate,
            release_deadline_ms=self.config.expert_release_deadline_ms,
        )
        return self.plan(
            envelope,
            byte_hit_rate=self.config.expert_assumed_byte_hit_rate,
            false_prefetch_ratio=self.config.expert_assumed_false_prefetch_ratio,
            workload=workload.value,
            evidence="hypothetical_control_input_not_model_measurement",
            capacity_scope=capacity_scope,
        )

    def plan(
        self,
        envelope: ResourceEnvelope,
        byte_hit_rate: float,
        false_prefetch_ratio: float = 0.0,
        workload: str = "manual",
        evidence: str = "manual_control_input_not_model_measurement",
        capacity_scope: str = "manual_envelope",
    ) -> dict[str, Any]:
        with self._lock:
            if self._planner is None or self._catalog is None:
                raise RuntimeError(self._error or "expert catalog is unavailable")
            false_prefetch_bytes = (
                self._catalog.active_expert_bytes_per_token
                * max(0.0, false_prefetch_ratio)
            )
            plan = self._planner.plan(
                envelope,
                byte_hit_rate=byte_hit_rate,
                false_prefetch_bytes_per_token=false_prefetch_bytes,
            ).to_dict()
            plan.update(
                {
                    "workload": workload,
                    "evidence": evidence,
                    "capacity_scope": capacity_scope,
                    "data_plane_ready": False,
                    "executable": False,
                    "envelope": {
                        "total_memory_gib": envelope.total_memory_gib,
                        "foreground_reserve_gib": envelope.foreground_reserve_gib,
                        "system_reserve_gib": envelope.system_reserve_gib,
                        "io_budget_gib_s": envelope.io_budget_gib_s,
                        "compute_duty_cycle": envelope.compute_duty_cycle,
                        "requested_token_rate": envelope.requested_token_rate,
                        "minimum_token_rate": envelope.minimum_token_rate,
                        "release_deadline_ms": envelope.release_deadline_ms,
                    },
                    "assumptions": {
                        "byte_hit_rate": byte_hit_rate,
                        "false_prefetch_ratio": false_prefetch_ratio,
                    },
                }
            )
            capacity_key = (
                plan.get("action"),
                plan.get("slots_per_layer"),
                plan.get("projected_reclaim_gib"),
                workload,
                capacity_scope,
                envelope.total_memory_gib,
                envelope.foreground_reserve_gib,
                envelope.system_reserve_gib,
                envelope.io_budget_gib_s,
                envelope.compute_duty_cycle,
            )
            if capacity_key != self._capacity_key:
                self._capacity_generation += 1
                self._capacity_key = capacity_key
            plan["generation"] = self._capacity_generation
            self._last_plan = plan
            self._last_updated_at = time.time()
            return self.status()

    def plan_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        envelope_payload = payload.get("envelope") or {}
        if not isinstance(envelope_payload, dict):
            raise ValueError("envelope must be an object")
        allowed = ResourceEnvelope.__dataclass_fields__
        envelope = ResourceEnvelope(
            **{
                key: float(value)
                for key, value in envelope_payload.items()
                if key in allowed
            }
        )
        return self.plan(
            envelope,
            byte_hit_rate=float(
                payload.get("byte_hit_rate", self.config.expert_assumed_byte_hit_rate)
            ),
            false_prefetch_ratio=float(
                payload.get(
                    "false_prefetch_ratio",
                    self.config.expert_assumed_false_prefetch_ratio,
                )
            ),
            workload=str(payload.get("workload", "manual")),
            capacity_scope=str(payload.get("capacity_scope", "manual_envelope")),
        )
