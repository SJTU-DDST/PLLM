from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .config import PLLMConfig
from .expert_catalog import ExpertCatalog
from .expert_residency import ResidencyPlanner, ResourceEnvelope
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
        self._load_catalog()

    def _load_catalog(self) -> None:
        try:
            catalog = ExpertCatalog.from_model(Path(self.config.model_path))
            self._catalog = catalog
            self._planner = ResidencyPlanner(catalog)
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
                "updated_at": self._last_updated_at,
                "guardrail": "recommendation_only_no_vllm_weight_mutation",
            }

    def recommend(
        self, snapshot: SensorSnapshot, workload: WorkloadClass
    ) -> dict[str, Any]:
        values = WORKLOAD_ENVELOPES[workload]
        if not snapshot.uma and snapshot.gpu_memory_total_gb:
            total_memory = snapshot.gpu_memory_total_gb
            capacity_scope = "discrete_gpu_vram"
        else:
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
