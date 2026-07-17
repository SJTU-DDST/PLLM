from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ControllerState(StrEnum):
    ACTIVE = "active"
    ELASTIC_RESIDENT = "elastic_resident"
    YIELDING = "yielding"
    QUIESCING = "quiescing"
    HIBERNATED = "hibernated"
    RESTORING = "restoring"
    # Kept for API compatibility with the first prototype.
    HOT_SLEEP = "hot_sleep"
    COLD_SLEEP = "cold_sleep"
    WAKING = "waking"
    ERROR = "error"


class PolicyMode(StrEnum):
    AUTO = "auto"
    AI_PRIORITY = "ai_priority"
    FOREGROUND_PRIORITY = "foreground_priority"
    KEEP_SLEEPING = "keep_sleeping"


class WorkloadClass(StrEnum):
    IDLE = "idle"
    GAME = "game"
    CREATIVE = "creative"
    GPU_PRESSURE = "gpu_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    POWER_PRESSURE = "power_pressure"


@dataclass(slots=True)
class ForegroundApp:
    pid: int = 0
    app_id: str = ""
    title: str = ""
    wm_class: str = ""
    available: bool = False

    @property
    def search_text(self) -> str:
        return " ".join((self.app_id, self.title, self.wm_class)).lower()


@dataclass(slots=True)
class ProcessGpuUsage:
    pid: int
    name: str = ""
    memory_gb: float = 0.0
    sm_util: int = 0
    memory_util: int = 0
    encoder_util: int = 0
    decoder_util: int = 0


@dataclass(slots=True)
class SensorSnapshot:
    timestamp: float
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_util: int = 0
    gpu_memory_total_gb: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_free_gb: float | None = None
    power_watts: float | None = None
    power_limit_watts: float | None = None
    temperature_c: int | None = None
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    swap_used_gb: float = 0.0
    memory_psi_some: float = 0.0
    memory_psi_full: float = 0.0
    cpu_percent: float = 0.0
    load_average: float = 0.0
    on_battery: bool = False
    battery_percent: float | None = None
    power_profile: str = "unknown"
    uma: bool = False
    foreground: ForegroundApp = field(default_factory=ForegroundApp)
    processes: list[ProcessGpuUsage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VLLMService:
    service_id: str
    base_url: str
    pid: int | None = None
    related_pids: list[int] = field(default_factory=list)
    command: str = ""
    model: str = ""
    healthy: bool = False
    controllable: bool = False
    sleeping: bool = False
    managed: bool = False
    last_error: str = ""
    last_sleep_level: int | None = None
    last_pause_mode: str = "keep"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Decision:
    action: str = "none"
    reason: str = ""
    workload: WorkloadClass = WorkloadClass.IDLE
    sleep_level: int | None = None
    pause_mode: str = "keep"
    score: float | None = None
    costs: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeStatus:
    state: ControllerState = ControllerState.ACTIVE
    mode: PolicyMode = PolicyMode.AUTO
    reason: str = ""
    workload: WorkloadClass = WorkloadClass.IDLE
    last_transition_at: float = 0.0
    last_action_duration_ms: float | None = None
    reclaimed_gb: float | None = None
    sleep_level: int | None = None
    pause_mode: str = "keep"
    transition_progress: float = 0.0
    restore_source: str = "local_nvme"
    decision: dict[str, Any] = field(default_factory=dict)
    hibercache: dict[str, Any] = field(default_factory=dict)
    expert_residency: dict[str, Any] = field(default_factory=dict)
    sensor: SensorSnapshot | None = None
    services: list[VLLMService] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["mode"] = self.mode.value
        data["workload"] = self.workload.value
        return data
