from __future__ import annotations

from pllm.config import PLLMConfig
from pllm.models import (
    ControllerState,
    ForegroundApp,
    ProcessGpuUsage,
    SensorSnapshot,
    VLLMService,
)
from pllm.policy import PolicyEngine


def snapshot(**overrides) -> SensorSnapshot:
    values = {
        "timestamp": 1.0,
        "gpu_available": True,
        "gpu_util": 0,
        "gpu_memory_total_gb": 96.0,
        "gpu_memory_used_gb": 60.0,
        "gpu_memory_free_gb": 36.0,
        "memory_total_gb": 256.0,
        "memory_available_gb": 220.0,
        "foreground": ForegroundApp(available=True),
    }
    values.update(overrides)
    return SensorSnapshot(**values)


def service() -> VLLMService:
    return VLLMService(
        service_id="vllm", base_url="http://127.0.0.1:8000", controllable=True
    )


def test_game_preempts_immediately_with_deep_hibernation() -> None:
    config = PLLMConfig()
    policy = PolicyEngine(config, model_size_gb=75.0)
    current = snapshot(
        foreground=ForegroundApp(
            pid=42, app_id="steam_app_123", wm_class="steam_app_123", available=True
        )
    )

    decision = policy.evaluate(current, [service()], ControllerState.ACTIVE, now=10.0)

    assert decision.action == "hibernate"
    assert decision.sleep_level == 2
    assert decision.pause_mode == "keep"
    assert decision.workload.value == "game"


def test_creative_workload_observes_hold_period() -> None:
    config = PLLMConfig(creative_hold_seconds=0.5)
    policy = PolicyEngine(config, model_size_gb=10.0)
    current = snapshot(
        gpu_util=20,
        foreground=ForegroundApp(
            pid=99, app_id="blender.desktop", wm_class="Blender", available=True
        ),
        processes=[ProcessGpuUsage(pid=99, sm_util=20)],
    )

    first = policy.evaluate(current, [service()], ControllerState.ACTIVE, now=10.0)
    second = policy.evaluate(current, [service()], ControllerState.ACTIVE, now=10.6)

    assert first.action == "none"
    assert second.action == "yield"
    assert second.sleep_level == 0


def test_foreground_priority_escalates_sustained_creative_yield() -> None:
    config = PLLMConfig(mode="foreground_priority", creative_hold_seconds=0.5)
    policy = PolicyEngine(config, model_size_gb=75.0)
    current = snapshot(
        gpu_util=60,
        foreground=ForegroundApp(
            pid=99, app_id="blender.desktop", wm_class="Blender", available=True
        ),
        processes=[ProcessGpuUsage(pid=99, sm_util=60)],
    )

    assert policy.evaluate(
        current, [service()], ControllerState.ACTIVE, now=10.0
    ).action == "none"
    yielded = policy.evaluate(
        current, [service()], ControllerState.ACTIVE, now=10.6
    )
    hibernated = policy.evaluate(
        current, [service()], ControllerState.YIELDING, now=11.1
    )

    assert yielded.action == "yield"
    assert hibernated.action == "hibernate"
    assert hibernated.sleep_level == 1
    assert "foreground-priority" in hibernated.reason


def test_saturated_creative_workload_discards_instead_of_copying_weights() -> None:
    config = PLLMConfig(mode="foreground_priority", creative_hold_seconds=0.5)
    policy = PolicyEngine(config, model_size_gb=75.0)
    current = snapshot(
        gpu_util=99,
        foreground=ForegroundApp(
            pid=99, app_id="blender.desktop", wm_class="Blender", available=True
        ),
        processes=[ProcessGpuUsage(pid=99, sm_util=99)],
    )

    assert policy.evaluate(
        current, [service()], ControllerState.ACTIVE, now=10.0
    ).action == "none"
    hibernated = policy.evaluate(
        current, [service()], ControllerState.ACTIVE, now=10.6
    )

    assert hibernated.action == "hibernate"
    assert hibernated.sleep_level == 2


def test_uma_always_uses_deep_sleep() -> None:
    policy = PolicyEngine(PLLMConfig(), model_size_gb=75.0)
    current = snapshot(
        uma=True,
        foreground=ForegroundApp(app_id="steam_app_1", available=True),
    )

    decision = policy.evaluate(current, [service()], ControllerState.ACTIVE, now=1.0)

    assert decision.sleep_level == 2


def test_yield_escalates_when_memory_pressure_appears() -> None:
    policy = PolicyEngine(PLLMConfig(min_available_memory_gb=20.0), model_size_gb=75.0)
    current = snapshot(memory_available_gb=4.0, uma=True)

    decision = policy.evaluate(current, [service()], ControllerState.YIELDING, now=3.0)

    assert decision.action == "hibernate"
    assert decision.sleep_level == 2


def test_sleeping_service_wakes_after_stable_idle() -> None:
    config = PLLMConfig(resume_idle_seconds=2.0)
    policy = PolicyEngine(config)
    current = snapshot()

    first = policy.evaluate(current, [service()], ControllerState.HOT_SLEEP, now=5.0)
    second = policy.evaluate(current, [service()], ControllerState.HOT_SLEEP, now=7.1)

    assert first.action == "none"
    assert second.action == "wake"


def test_ai_priority_ignores_noncritical_gpu_pressure() -> None:
    config = PLLMConfig(mode="ai_priority", external_gpu_pressure_percent=50)
    policy = PolicyEngine(config)
    current = snapshot(processes=[ProcessGpuUsage(pid=500, sm_util=90)])

    decision = policy.evaluate(current, [service()], ControllerState.ACTIVE, now=2.0)

    assert decision.action == "none"
    assert "AI priority" in decision.reason


def test_vllm_worker_process_does_not_trigger_itself() -> None:
    config = PLLMConfig(external_gpu_pressure_percent=50)
    policy = PolicyEngine(config)
    current = snapshot(processes=[ProcessGpuUsage(pid=501, sm_util=99)])
    current_service = service()
    current_service.pid = 500
    current_service.related_pids = [500, 501]

    decision = policy.evaluate(
        current, [current_service], ControllerState.ACTIVE, now=2.0
    )

    assert decision.action == "none"
    assert decision.workload.value == "idle"
