from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PLLMConfig
from .models import SensorSnapshot, WorkloadClass


@dataclass(slots=True)
class CalibrationProfile:
    yield_latency_ms: float = 40.0
    hibernate_latency_ms: float = 1800.0
    local_restore_ms: float = 18_000.0
    remote_restore_ms: float = 9_000.0
    hibernate_reclaim_ratio: float = 0.95
    foreground_duration_seconds: float = 300.0
    sample_count: int = 0

    @classmethod
    def load(cls, path: Path) -> "CalibrationProfile":
        if not path.exists():
            return cls()
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            known = cls.__dataclass_fields__
            return cls(**{key: values[key] for key in known if key in values})
        except (OSError, ValueError, TypeError):
            return cls()

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class CostPlan:
    action: str
    level: int
    score: float
    costs: dict[str, float]
    reason: str


class ForegroundCostModel:
    """Small, deterministic planner calibrated with measurements from this host."""

    def __init__(self, config: PLLMConfig, model_size_gb: float) -> None:
        default_path = Path.home() / ".local" / "share" / "pllm" / "calibration.json"
        path = Path(config.calibration_path).expanduser() if config.calibration_path else default_path
        self.config = config
        self.model_size_gb = model_size_gb
        self.profile = CalibrationProfile.load(path)

    def plan(
        self, snapshot: SensorSnapshot, workload: WorkloadClass, reason: str
    ) -> CostPlan:
        pressure = self._pressure(snapshot)
        memory_deficit = self._memory_deficit(snapshot)
        duration = max(
            1.0,
            self.profile.foreground_duration_seconds
            if self.profile.sample_count
            else self.config.foreground_duration_seconds,
        )
        restore_source = (
            self.profile.remote_restore_ms
            if self.config.rdma_peer and self.config.remote_weight_source_enabled
            else self.profile.local_restore_ms
        )

        # Scores are normalized penalties. Hard constraints are represented by
        # a large infeasibility penalty rather than a hidden branch.
        yield_infeasible = workload in {
            WorkloadClass.GAME,
            WorkloadClass.MEMORY_PRESSURE,
            WorkloadClass.POWER_PRESSURE,
        }
        yield_score = (
            self.profile.yield_latency_ms / 500.0
            + pressure * 0.8
            + memory_deficit * 8.0
            + (100.0 if yield_infeasible else 0.0)
        )
        restore_penalty = restore_source / (duration * 1000.0)
        hibernate_score = (
            self.profile.hibernate_latency_ms / 3000.0
            + restore_penalty
            + max(0.0, 0.95 - self.profile.hibernate_reclaim_ratio) * 10.0
        )
        costs = {
            "yield": round(yield_score, 4),
            "hibernate": round(hibernate_score, 4),
            "pressure": round(pressure, 4),
            "memory_deficit": round(memory_deficit, 4),
            "restore_penalty": round(restore_penalty, 4),
        }

        if yield_score < hibernate_score:
            return CostPlan(
                action="yield",
                level=0,
                score=yield_score,
                costs=costs,
                reason=f"QoS micro-yield selected: {reason}",
            )
        return CostPlan(
            action="hibernate",
            level=(
                2
                if snapshot.uma
                or workload
                in {
                    WorkloadClass.MEMORY_PRESSURE,
                    WorkloadClass.POWER_PRESSURE,
                    WorkloadClass.GAME,
                }
                or (workload == WorkloadClass.CREATIVE and pressure >= 0.9)
                else self._discrete_level(snapshot)
            ),
            score=hibernate_score,
            costs=costs,
            reason=f"resource hibernation selected: {reason}",
        )

    def _pressure(self, snapshot: SensorSnapshot) -> float:
        external = max(
            (
                max(item.sm_util, item.encoder_util, item.decoder_util)
                for item in snapshot.processes
            ),
            default=snapshot.gpu_util,
        )
        return min(1.0, max(0.0, external / 100.0))

    def _memory_deficit(self, snapshot: SensorSnapshot) -> float:
        system = max(
            0.0,
            (self.config.min_available_memory_gb - snapshot.memory_available_gb)
            / max(1.0, self.config.min_available_memory_gb),
        )
        if snapshot.gpu_memory_free_gb is None:
            return system
        device = max(
            0.0,
            (self.config.min_free_vram_gb - snapshot.gpu_memory_free_gb)
            / max(1.0, self.config.min_free_vram_gb),
        )
        return max(system, device)

    def _discrete_level(self, snapshot: SensorSnapshot) -> int:
        required = self.model_size_gb * 1.15 + self.config.hot_sleep_memory_reserve_gb
        return 1 if snapshot.memory_available_gb >= required else 2
