from __future__ import annotations

import time
from collections.abc import Iterable

from .config import PLLMConfig
from .cost_model import ForegroundCostModel
from .models import (
    ControllerState,
    Decision,
    PolicyMode,
    SensorSnapshot,
    VLLMService,
    WorkloadClass,
)


SLEEPING_STATES = {
    ControllerState.YIELDING,
    ControllerState.HIBERNATED,
    ControllerState.HOT_SLEEP,
    ControllerState.COLD_SLEEP,
}


class PolicyEngine:
    def __init__(self, config: PLLMConfig, model_size_gb: float = 0.0) -> None:
        self.config = config
        self.model_size_gb = model_size_gb
        self.cost_model = ForegroundCostModel(config, model_size_gb)
        self._pressure_since: float | None = None
        self._safe_since: float | None = None
        self._last_wake_at: float | None = None

    def evaluate(
        self,
        snapshot: SensorSnapshot,
        services: Iterable[VLLMService],
        state: ControllerState,
        now: float | None = None,
    ) -> Decision:
        now = time.monotonic() if now is None else now
        mode = PolicyMode(self.config.mode)
        controllable = [service for service in services if service.controllable]
        workload, reason, immediate = self.classify(snapshot, controllable)

        if mode == PolicyMode.KEEP_SLEEPING:
            if state not in SLEEPING_STATES and controllable:
                return Decision("hibernate", "keep-sleeping mode", workload, 2)
            return Decision(workload=workload, reason=reason)

        if state in SLEEPING_STATES:
            if state == ControllerState.YIELDING and workload in {
                WorkloadClass.GAME,
                WorkloadClass.MEMORY_PRESSURE,
                WorkloadClass.POWER_PRESSURE,
            }:
                plan = self.cost_model.plan(snapshot, workload, reason)
                return Decision(
                    "hibernate",
                    f"micro-yield escalated: {reason}",
                    workload,
                    2,
                    "keep",
                    plan.score,
                    plan.costs,
                )
            if (
                state == ControllerState.YIELDING
                and mode == PolicyMode.FOREGROUND_PRIORITY
                and workload == WorkloadClass.CREATIVE
                and self._pressure_since is not None
                and now - self._pressure_since
                >= max(0.0, self.config.creative_hold_seconds * 2)
            ):
                level = self.choose_sleep_level(snapshot, workload)
                return Decision(
                    "hibernate",
                    f"foreground-priority yield escalated: {reason}",
                    workload,
                    level,
                    "keep",
                )
            if workload != WorkloadClass.IDLE:
                self._safe_since = None
                return Decision(workload=workload, reason=reason)
            if self._safe_since is None:
                self._safe_since = now
            idle_target = (
                self.config.yield_resume_idle_seconds
                if state == ControllerState.YIELDING
                else self.config.resume_idle_seconds
            )
            if mode != PolicyMode.FOREGROUND_PRIORITY and (
                now - self._safe_since >= idle_target
            ):
                return Decision("wake", "foreground pressure cleared", workload)
            if mode == PolicyMode.FOREGROUND_PRIORITY and (
                now - self._safe_since >= idle_target
            ):
                return Decision("wake", "foreground workload is idle", workload)
            return Decision(workload=workload, reason="waiting for stable idle period")

        self._safe_since = None
        if workload == WorkloadClass.IDLE or not controllable:
            self._pressure_since = None
            return Decision(workload=workload, reason=reason)

        if mode == PolicyMode.AI_PRIORITY and workload not in {
            WorkloadClass.MEMORY_PRESSURE,
            WorkloadClass.POWER_PRESSURE,
        }:
            return Decision(workload=workload, reason=f"AI priority ignored: {reason}")

        if (
            not immediate
            and self._last_wake_at is not None
            and now - self._last_wake_at < self.config.wake_cooldown_seconds
        ):
            return Decision(workload=workload, reason=f"wake cooldown: {reason}")

        if self._pressure_since is None:
            self._pressure_since = now
        hold = 0.0 if immediate else self.config.creative_hold_seconds
        if now - self._pressure_since < hold:
            return Decision(workload=workload, reason=f"confirming pressure: {reason}")

        plan = self.cost_model.plan(snapshot, workload, reason)
        return Decision(
            plan.action,
            plan.reason,
            workload,
            plan.level,
            "keep",
            plan.score,
            plan.costs,
        )

    def classify(
        self,
        snapshot: SensorSnapshot,
        services: Iterable[VLLMService],
    ) -> tuple[WorkloadClass, str, bool]:
        text = snapshot.foreground.search_text
        if any(pattern.lower() in text for pattern in self.config.game_patterns):
            return WorkloadClass.GAME, f"game in foreground: {text.strip()}", True

        if snapshot.memory_available_gb < self.config.min_available_memory_gb:
            return (
                WorkloadClass.MEMORY_PRESSURE,
                f"available memory {snapshot.memory_available_gb:.1f} GiB",
                True,
            )
        if snapshot.memory_psi_full >= 0.1 or snapshot.memory_psi_some >= 1.0:
            return WorkloadClass.MEMORY_PRESSURE, "system memory PSI pressure", True
        if (
            snapshot.gpu_memory_free_gb is not None
            and snapshot.gpu_memory_free_gb < self.config.min_free_vram_gb
        ):
            return (
                WorkloadClass.MEMORY_PRESSURE,
                f"free VRAM {snapshot.gpu_memory_free_gb:.1f} GiB",
                True,
            )
        if snapshot.on_battery and (
            snapshot.battery_percent is None
            or snapshot.battery_percent <= self.config.low_battery_percent
        ):
            return WorkloadClass.POWER_PRESSURE, "battery power policy", True

        service_pids = {service.pid for service in services if service.pid}
        for service in services:
            service_pids.update(service.related_pids)
        foreground_usage = next(
            (item for item in snapshot.processes if item.pid == snapshot.foreground.pid),
            None,
        )
        creative = any(
            pattern.lower() in text for pattern in self.config.creative_patterns
        )
        if creative:
            active = (
                max(
                    foreground_usage.sm_util,
                    foreground_usage.encoder_util,
                    foreground_usage.decoder_util,
                )
                if foreground_usage
                else 0
            )
            if active >= self.config.creative_gpu_percent:
                return (
                    WorkloadClass.CREATIVE,
                    f"creative foreground GPU activity {active}%",
                    False,
                )

        external_util = max(
            (
                max(item.sm_util, item.encoder_util, item.decoder_util)
                for item in snapshot.processes
                if item.pid not in service_pids
            ),
            default=0,
        )
        threshold = min(
            self.config.external_gpu_pressure_percent,
            self.config.soft_gpu_pressure_percent,
        )
        if external_util >= threshold:
            return (
                WorkloadClass.GPU_PRESSURE,
                f"non-vLLM GPU activity {external_util}%",
                False,
            )
        return WorkloadClass.IDLE, "no foreground pressure", False

    def choose_sleep_level(
        self, snapshot: SensorSnapshot, workload: WorkloadClass
    ) -> int:
        if snapshot.uma or snapshot.on_battery:
            return 2
        if workload in {WorkloadClass.MEMORY_PRESSURE, WorkloadClass.POWER_PRESSURE}:
            return 2
        if workload == WorkloadClass.CREATIVE:
            foreground_usage = next(
                (
                    item
                    for item in snapshot.processes
                    if item.pid == snapshot.foreground.pid
                ),
                None,
            )
            if foreground_usage is not None and max(
                foreground_usage.sm_util,
                foreground_usage.encoder_util,
                foreground_usage.decoder_util,
            ) >= 90:
                return 2
        required = self.model_size_gb * 1.15 + self.config.hot_sleep_memory_reserve_gb
        return 1 if snapshot.memory_available_gb >= required else 2

    def mark_wake(self, now: float | None = None) -> None:
        self._last_wake_at = time.monotonic() if now is None else now
        self._pressure_since = None
        self._safe_since = None
