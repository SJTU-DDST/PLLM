from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from .config import PLLMConfig
from .benchmarks import run_safe_benchmark
from .capabilities import CapabilityProbe
from .foreground import ForegroundProvider
from .expert_control import ExpertResidencyControlPlane
from .expert_runtime_client import ExpertRuntimeClient
from .hibercache import HiberCacheManager
from .models import (
    ControllerState,
    PolicyMode,
    RuntimeStatus,
    WorkloadClass,
)
from .monitor import SystemMonitor
from .policy import PolicyEngine, SLEEPING_STATES
from .storage import Storage
from .vllm import VLLMClient, VLLMDiscovery, VLLMManager


LOGGER = logging.getLogger(__name__)


class PLLMController:
    def __init__(
        self,
        config: PLLMConfig,
        storage: Storage,
        monitor: SystemMonitor | None = None,
        manager: VLLMManager | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        foreground = ForegroundProvider(config.foreground_file)
        self.monitor = monitor or SystemMonitor(foreground)
        if manager is None:
            client = VLLMClient(config.request_timeout_seconds)
            discovery = VLLMDiscovery(
                client,
                config.default_vllm_urls,
                config.excluded_process_patterns,
            )
            manager = VLLMManager(client, discovery)
        self.manager = manager
        self.hibercache = HiberCacheManager(config)
        self.expert_residency = ExpertResidencyControlPlane(config)
        self.expert_runtime = ExpertRuntimeClient(
            str(config.resolved_expert_runtime_socket())
        )
        self.capability_probe = CapabilityProbe(config, self.hibercache)
        self.policy = PolicyEngine(config, _model_size_gb(Path(config.model_path)))
        self._status = RuntimeStatus(
            mode=PolicyMode(config.mode), last_transition_at=time.time()
        )
        self._status_lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_service_refresh = 0.0
        self._last_expert_resize_at = 0.0
        self._inference_phases: dict[str, str] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="pllm-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.monitor.close()

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            self._status.hibercache = self.hibercache.status()
            self._status.expert_residency = self._expert_status()
            return self._status.to_dict()

    def capabilities(self, refresh: bool = False) -> dict[str, Any]:
        result = self.capability_probe.collect(refresh=refresh)
        result["expert_residency"] = self._expert_status()
        return result

    def expert_residency_status(self) -> dict[str, Any]:
        return self._expert_status()

    def expert_dataplane_status(self) -> dict[str, Any]:
        return self.expert_runtime.status()

    def plan_expert_residency(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.expert_residency.plan_from_payload(payload)
        result = self._expert_status()
        self.storage.add_event(
            "expert_plan",
            self._status.state.value,
            result.get("plan", {}).get("reason", "expert residency plan"),
            {
                "evidence": result.get("plan", {}).get("evidence"),
                "action": result.get("plan", {}).get("action"),
                "data_plane_ready": result.get("data_plane_ready", False),
            },
        )
        return result

    def expert_dataplane_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.expert_data_plane_enabled:
            raise RuntimeError("expert data plane is disabled")
        action = str(payload.get("action", "status")).strip().lower()
        if action == "status":
            return self.expert_runtime.status()
        if self.config.dry_run:
            raise RuntimeError("dry-run mode cannot mutate the expert data plane")
        if action == "set_phase":
            result = self.expert_runtime.set_phase(
                str(payload.get("phase", "idle")),
                bool(payload.get("reset_decode", False)),
            )
            return result
        if action not in {"prefetch", "evict", "resize", "evict_all"}:
            raise ValueError(f"unknown expert data-plane action: {action}")

        with self._transition_lock:
            try:
                self._set_state(
                    ControllerState.QUIESCING,
                    f"expert data-plane {action}",
                    progress=0.15,
                    sleep_level=0,
                    pause_mode="keep",
                )
                if not self.config.dry_run:
                    self.manager.sleep_all(0, mode="keep")
                if action == "resize":
                    slots = int(payload.get("slots_per_layer", 0))
                    if slots < 22:
                        raise ValueError("slots_per_layer cannot be below Top-22")
                    if payload.get("phase"):
                        self.expert_runtime.set_phase(str(payload["phase"]))
                    result = self.expert_runtime.resize(
                        slots,
                        retain_policy=str(payload.get("retain_policy", "lru")),
                    )
                elif action == "prefetch":
                    result = self.expert_runtime.prefetch(
                        int(payload["layer"]),
                        [int(item) for item in payload.get("experts", [])],
                    )
                elif action == "evict":
                    result = self.expert_runtime.evict(
                        int(payload["layer"]),
                        [int(item) for item in payload.get("experts", [])],
                    )
                else:
                    result = self.expert_runtime.evict_all()
                if not self.config.dry_run and action != "evict_all":
                    self.manager.wake_all()
            except Exception:
                self._set_state(
                    ControllerState.ERROR,
                    f"expert data-plane {action} failed; vLLM remains quiesced",
                )
                raise
            if action != "evict_all":
                full_slots = int(
                    self.expert_residency.status()
                    .get("model", {})
                    .get("experts_per_layer", 0)
                )
                runtime_slots = int(
                    self.expert_runtime.status().get("slots_per_layer", 0)
                )
                target_state = (
                    ControllerState.ACTIVE
                    if runtime_slots >= full_slots > 0
                    else ControllerState.ELASTIC_RESIDENT
                )
                self._set_state(
                    target_state,
                    (
                        f"expert data-plane resized to {slots} slots/layer"
                        if action == "resize"
                        else f"expert data-plane {action} completed"
                    ),
                    progress=1.0,
                    sleep_level=None,
                    pause_mode="keep",
                )
                if action == "resize":
                    self._last_expert_resize_at = time.monotonic()
            else:
                self._set_state(
                    ControllerState.QUIESCING,
                    "expert mappings evicted; vLLM remains quiesced",
                    progress=1.0,
                    sleep_level=0,
                    pause_mode="keep",
                )
            self.storage.add_event(
                "expert_dataplane",
                self._status.state.value,
                action,
                result,
            )
            return result

    def services(self) -> list[dict[str, Any]]:
        with self._status_lock:
            return [service.to_dict() for service in self._status.services]

    def update_policy(self, values: dict[str, Any]) -> dict[str, Any]:
        self.config.update(values)
        self.config.save()
        with self._status_lock:
            self._status.mode = PolicyMode(self.config.mode)
        self.storage.add_event(
            "policy", self._status.state.value, "policy updated", values
        )
        return self.config.public_dict()

    def action(self, action: str, level: int | None = None) -> dict[str, Any]:
        normalized = action.strip().lower()
        if normalized == "yield":
            if self._status.state == ControllerState.YIELDING:
                return self.status()
            self._sleep(0, "manual QoS yield", mode="keep")
        elif normalized in {"pause", "hibernate"}:
            if self._status.state in SLEEPING_STATES:
                if (
                    normalized == "hibernate"
                    and self._status.state == ControllerState.YIELDING
                ):
                    self._sleep(level or self._recommended_level(), "manual hibernation", mode="keep")
                return self.status()
            self._sleep(
                level or self._recommended_level(),
                "manual hibernation",
                mode="keep",
            )
        elif normalized == "wake":
            if self._status.state == ControllerState.ACTIVE:
                return self.status()
            self._wake("manual wake")
        elif normalized == "auto":
            self.update_policy({"mode": PolicyMode.AUTO.value})
        elif normalized == "snooze":
            self.update_policy({"mode": PolicyMode.KEEP_SLEEPING.value})
            if self._status.state not in SLEEPING_STATES:
                self._sleep(level or 2, "manual keep sleeping", mode="keep")
        elif normalized == "benchmark":
            metrics = run_safe_benchmark(self.config.model_path)
            experiment_id = self.storage.add_experiment(
                "safe_probe", "CPU and local model storage", metrics
            )
            self.storage.add_event(
                "benchmark", self._status.state.value, "safe benchmark completed", metrics
            )
            return {"experiment_id": experiment_id, "metrics": metrics}
        else:
            raise ValueError(f"Unknown action: {action}")
        return self.status()

    def can_proxy(self) -> bool:
        with self._status_lock:
            return self._status.state in {
                ControllerState.ACTIVE,
                ControllerState.ELASTIC_RESIDENT,
                ControllerState.YIELDING,
            }

    def compile_policy(self, text: str, apply: bool = False) -> dict[str, Any]:
        normalized = text.strip().lower()
        if not normalized:
            raise ValueError("Policy text is required")
        values: dict[str, Any] = {}
        rules = []
        if any(token in normalized for token in ("blender", "渲染", "剪辑", "创作")):
            rules.append({"workload": "creative", "action": "yield_then_hibernate"})
            values["creative_hold_seconds"] = 0.5
        if any(token in normalized for token in ("游戏", "game", "steam")):
            rules.append({"workload": "game", "action": "hibernate", "level": 2})
        if any(token in normalized for token in ("电池", "battery", "省电")):
            rules.append({"workload": "power_pressure", "action": "hibernate", "level": 2})
        if any(token in normalized for token in ("立即", "300ms", "500ms")):
            values["creative_hold_seconds"] = 0.25
        if not rules:
            rules.append({"workload": "gpu_pressure", "action": "yield"})
        if apply and values:
            self.update_policy(values)
        result = {
            "input": text,
            "advisor": "deterministic_local_guard",
            "rules": rules,
            "config_patch": values,
            "applied": bool(apply and values),
            "safety": "validated; no process termination permission",
        }
        self.storage.add_event("policy_compile", self._status.state.value, text, result)
        return result

    def proxy_target(self) -> str | None:
        return self.manager.target_url()

    def mark_inference_phase(
        self,
        phase: str,
        reset_decode: bool = False,
        request_id: str = "",
    ) -> None:
        normalized = phase.strip().lower()
        if normalized not in {"idle", "prefill", "decode"}:
            raise ValueError(f"invalid inference phase: {phase}")
        effective_reset = reset_decode
        if request_id:
            with self._status_lock:
                was_idle = not self._inference_phases
                if normalized == "idle":
                    self._inference_phases.pop(request_id, None)
                else:
                    self._inference_phases[request_id] = normalized
                phases = set(self._inference_phases.values())
                normalized = (
                    "prefill"
                    if "prefill" in phases
                    else "decode"
                    if "decode" in phases
                    else "idle"
                )
                effective_reset = reset_decode and was_idle
        runtime = self.expert_runtime.status()
        if not runtime.get("online"):
            return
        try:
            self.expert_runtime.set_phase(
                normalized, reset_decode=effective_reset
            )
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.debug("Could not update EER inference phase: %s", exc)

    def prepare_inference_request(self, request_id: str) -> None:
        """Restore full expert residency before forwarding a new prefill."""
        self.mark_inference_phase(
            "prefill", reset_decode=True, request_id=request_id
        )
        runtime = self.expert_runtime.status()
        if not (
            self.config.expert_data_plane_enabled
            and runtime.get("online")
            and runtime.get("data_plane_ready")
        ):
            return
        layers = runtime.get("data_plane", {}).get("layers", [])
        full_slots = max(
            (int(layer.get("global_experts", 0)) for layer in layers),
            default=0,
        )
        current_slots = int(runtime.get("slots_per_layer", 0))
        if full_slots > 0 and current_slots < full_slots:
            self.expert_dataplane_action(
                {
                    "action": "resize",
                    "slots_per_layer": full_slots,
                    "retain_policy": "lru",
                    "phase": "prefill",
                }
            )

    def refresh_services(self) -> list[dict[str, Any]]:
        services = self.manager.refresh()
        with self._status_lock:
            self._status.services = services
        self._last_service_refresh = time.monotonic()
        return [service.to_dict() for service in services]

    def _run(self) -> None:
        try:
            self.refresh_services()
        except Exception as exc:
            LOGGER.warning("Initial vLLM discovery failed: %s", exc)
        while not self._stop_event.is_set():
            loop_started = time.monotonic()
            try:
                snapshot = self.monitor.collect()
                if (
                    loop_started - self._last_service_refresh
                    >= self.config.service_refresh_seconds
                ):
                    self.refresh_services()
                with self._status_lock:
                    self._status.sensor = snapshot
                    services = list(self._status.services)
                    state = self._status.state
                decision = self.policy.evaluate(snapshot, services, state, loop_started)
                if self.config.expert_residency_enabled:
                    self.expert_residency.recommend(snapshot, decision.workload)
                expert_status = self._expert_status()
                with self._status_lock:
                    self._status.workload = decision.workload
                    self._status.expert_residency = expert_status
                    if decision.reason:
                        self._status.reason = decision.reason
                if decision.action != "none" or not self._status.decision:
                    with self._status_lock:
                        self._status.decision = {
                            "action": decision.action,
                            "reason": decision.reason,
                            "score": decision.score,
                            "costs": decision.costs,
                        }
                if self._maybe_auto_resize(expert_status):
                    continue
                if decision.action in {"yield", "hibernate", "sleep"}:
                    self._sleep(
                        decision.sleep_level if decision.sleep_level is not None else 2,
                        decision.reason,
                        mode=decision.pause_mode,
                    )
                elif decision.action == "wake":
                    self._wake(decision.reason)
            except Exception as exc:
                LOGGER.exception("PLLM monitor iteration failed")
                self._set_state(ControllerState.ERROR, str(exc))
                self.storage.add_event(
                    "error", ControllerState.ERROR.value, str(exc), {}
                )
            elapsed = time.monotonic() - loop_started
            self._stop_event.wait(max(0.01, self.config.poll_interval_seconds - elapsed))

    def _sleep(self, level: int, reason: str, mode: str = "keep") -> None:
        if not self._transition_lock.acquire(blocking=False):
            return
        started = time.monotonic()
        before = self._current_gpu_used()
        try:
            transition = (
                ControllerState.YIELDING if level == 0 else ControllerState.QUIESCING
            )
            self._set_state(transition, reason, progress=0.15, sleep_level=level, pause_mode=mode)
            self.storage.pause_running_replays()
            runtime_ready = False
            if level > 0:
                self.hibercache.enforce_quota()
                runtime = self.expert_runtime.status()
                runtime_ready = bool(
                    runtime.get("online") and runtime.get("data_plane_ready")
                )
                if runtime_ready and not self.config.dry_run:
                    self.manager.sleep_all(0, mode=mode)
                    self.expert_runtime.suspend()
            if self.config.dry_run:
                controlled = len(self.manager.controllable())
            elif level > 0 and runtime_ready:
                controlled = self.manager.deep_sleep_all_from_quiesced(
                    level, mode=mode
                )
            else:
                controlled = self.manager.sleep_all(level, mode=mode)
            after_snapshot = self.monitor.collect()
            with self._status_lock:
                self._status.sensor = after_snapshot
            after = after_snapshot.gpu_memory_used_gb
            reclaimed = (
                max(0.0, before - after)
                if before is not None and after is not None
                else None
            )
            state = (
                ControllerState.YIELDING
                if level == 0
                else ControllerState.HIBERNATED
            )
            duration = (time.monotonic() - started) * 1000
            self._set_state(
                state,
                reason,
                duration,
                reclaimed,
                progress=1.0,
                sleep_level=level,
                pause_mode=mode,
            )
            self.storage.add_event(
                "sleep",
                state.value,
                reason,
                {
                    "level": level,
                    "mode": mode,
                    "services": controlled,
                    "duration_ms": duration,
                    "reclaimed_gb": reclaimed,
                    "dry_run": self.config.dry_run,
                },
            )
        except Exception as exc:
            self._set_state(ControllerState.ERROR, str(exc))
            self.storage.add_event("sleep_error", "error", str(exc), {"level": level})
            raise
        finally:
            self._transition_lock.release()

    def _wake(self, reason: str) -> None:
        if not self._transition_lock.acquire(blocking=False):
            return
        started = time.monotonic()
        try:
            self._set_state(ControllerState.RESTORING, reason, progress=0.1)
            if self.config.dry_run:
                controlled = len(self.manager.controllable())
            else:
                controlled = self.manager.wake_all()
            runtime = self.expert_runtime.status()
            if runtime.get("online") and runtime.get("suspended"):
                self.expert_runtime.resume()
            self.storage.resume_paused_replays()
            duration = (time.monotonic() - started) * 1000
            self.policy.mark_wake()
            self._set_state(
                ControllerState.ACTIVE,
                reason,
                duration,
                None,
                progress=1.0,
                sleep_level=None,
            )
            self.storage.add_event(
                "wake",
                ControllerState.ACTIVE.value,
                reason,
                {
                    "services": controlled,
                    "duration_ms": duration,
                    "dry_run": self.config.dry_run,
                },
            )
        except Exception as exc:
            self._set_state(ControllerState.ERROR, str(exc))
            self.storage.add_event("wake_error", "error", str(exc), {})
            raise
        finally:
            self._transition_lock.release()

    def _set_state(
        self,
        state: ControllerState,
        reason: str,
        duration_ms: float | None = None,
        reclaimed_gb: float | None = None,
        progress: float | None = None,
        sleep_level: int | None = None,
        pause_mode: str | None = None,
    ) -> None:
        with self._status_lock:
            self._status.state = state
            self._status.reason = reason
            self._status.last_transition_at = time.time()
            self._status.last_action_duration_ms = duration_ms
            self._status.reclaimed_gb = reclaimed_gb
            if progress is not None:
                self._status.transition_progress = max(0.0, min(1.0, progress))
            self._status.sleep_level = sleep_level
            if pause_mode is not None:
                self._status.pause_mode = pause_mode

    def _recommended_level(self) -> int:
        with self._status_lock:
            snapshot = self._status.sensor
        if snapshot is None:
            return 2
        return self.policy.choose_sleep_level(snapshot, WorkloadClass.GPU_PRESSURE)

    def _current_gpu_used(self) -> float | None:
        with self._status_lock:
            return (
                self._status.sensor.gpu_memory_used_gb
                if self._status.sensor is not None
                else None
            )

    def _expert_status(self) -> dict[str, Any]:
        projection = self.expert_residency.status()
        runtime = self.expert_runtime.status()
        with self._status_lock:
            projection["active_inference_requests"] = len(self._inference_phases)
            projection["request_phases"] = dict(self._inference_phases)
        projection["decode_plan"] = self.expert_residency.plan_decode_residency(
            runtime
        )
        projection["data_plane"] = runtime
        if runtime.get("data_plane_ready"):
            projection["data_plane_ready"] = True
            projection["backend"] = runtime.get(
                "backend", "vllm_modelopt_nvfp4_marlin"
            )
            projection["evidence"] = "live_exact_expert_dataplane"
            projection["guardrail"] = "actual_topk_blocking_miss_exact_load"
            plan = dict(projection.get("plan") or {})
            if plan:
                plan["executable"] = True
                plan["data_plane_ready"] = True
                projection["plan"] = plan
        return projection

    def _maybe_auto_resize(self, expert_status: dict[str, Any]) -> bool:
        if not (
            self.config.expert_data_plane_enabled
            and self.config.expert_auto_resize_enabled
            and expert_status.get("data_plane_ready")
        ):
            return False
        capacity_plan = expert_status.get("plan") or {}
        decode_plan = expert_status.get("decode_plan") or {}
        if capacity_plan.get("action") not in {"elastic_resident", "full_resident"}:
            return False
        if capacity_plan.get("action") == "elastic_resident":
            if not self.config.decode_elastic_enabled:
                return False
            if decode_plan.get("action") != "decode_elastic":
                return False
            desired = int(decode_plan.get("slots_per_layer", 0))
            retain_policy = "decode_hot"
        else:
            desired = int(
                expert_status.get("model", {}).get("experts_per_layer", 0)
            )
            retain_policy = "lru"
        if desired < 22:
            return False
        layers = (
            expert_status.get("data_plane", {})
            .get("data_plane", {})
            .get("layers", [])
        )
        current = {int(item.get("slot_count", 0)) for item in layers}
        if current == {desired}:
            return False
        if (
            time.monotonic() - self._last_expert_resize_at
            < self.config.expert_resize_cooldown_seconds
        ):
            return False
        self.expert_dataplane_action(
            {
                "action": "resize",
                "slots_per_layer": desired,
                "retain_policy": retain_policy,
            }
        )
        return True


def _model_size_gb(model_path: Path) -> float:
    if not model_path.exists():
        return 0.0
    try:
        total = sum(
            path.stat().st_size for path in model_path.glob("*.safetensors") if path.is_file()
        )
        return total / 1024**3
    except OSError:
        return 0.0
