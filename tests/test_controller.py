from __future__ import annotations

from pathlib import Path

from pllm.config import PLLMConfig
from pllm.controller import PLLMController
from pllm.models import SensorSnapshot, VLLMService
from pllm.storage import Storage


class FakeMonitor:
    def __init__(self) -> None:
        self.used = 60.0

    def collect(self) -> SensorSnapshot:
        self.used -= 1.0
        return SensorSnapshot(
            timestamp=1.0,
            gpu_available=True,
            gpu_memory_used_gb=self.used,
            gpu_memory_free_gb=96.0 - self.used,
            memory_total_gb=256.0,
            memory_available_gb=220.0,
        )

    def close(self) -> None:
        pass


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | str]] = []
        self.service = VLLMService(
            service_id="fake",
            base_url="http://127.0.0.1:18000",
            healthy=True,
            controllable=True,
        )

    def controllable(self):
        return [self.service]

    def sleep_all(self, level: int, mode: str = "keep") -> int:
        self.calls.append(("sleep", level))
        self.calls.append(("mode", mode))
        return 1

    def wake_all(self) -> int:
        self.calls.append(("wake", 0))
        return 1

    def target_url(self) -> str:
        return self.service.base_url


class FakeExpertRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def status(self):
        return {"online": True, "data_plane_ready": True}

    def set_phase(self, phase: str, reset_decode: bool = False):
        self.calls.append((phase, reset_decode))
        return {"phase": phase}


def test_manual_yield_escalation_and_wake(tmp_path: Path) -> None:
    manager = FakeManager()
    monitor = FakeMonitor()
    storage = Storage(tmp_path / "events.sqlite3")
    controller = PLLMController(
        PLLMConfig(
            hibercache_dir=str(tmp_path / "cache"),
            expert_runtime_socket=str(tmp_path / "pllm-eer.sock"),
        ),
        storage,
        monitor=monitor,
        manager=manager,
    )
    controller._status.sensor = monitor.collect()
    replay_id = storage.create_replay({"messages": [{"content": "code"}]}, "running")
    storage.update_replay_progress(replay_id, 12)

    yielded = controller.action("yield")
    paused_replay = storage.get_replay(replay_id)
    hibernated = controller.action("hibernate", level=2)
    active = controller.action("wake")

    assert yielded["state"] == "yielding"
    assert yielded["pause_mode"] == "keep"
    assert paused_replay is not None
    assert paused_replay["status"] == "paused"
    assert paused_replay["paused_at_token"] == 12
    assert hibernated["state"] == "hibernated"
    assert hibernated["sleep_level"] == 2
    assert active["state"] == "active"
    assert storage.get_replay(replay_id)["status"] == "running"
    assert manager.calls == [
        ("sleep", 0),
        ("mode", "keep"),
        ("sleep", 2),
        ("mode", "keep"),
        ("wake", 0),
    ]


def test_request_phase_aggregation_never_marks_busy_peer_idle(tmp_path: Path) -> None:
    controller = PLLMController(
        PLLMConfig(
            model_path=str(tmp_path / "missing-model"),
            hibercache_dir=str(tmp_path / "cache"),
            expert_runtime_socket=str(tmp_path / "pllm-eer.sock"),
        ),
        Storage(tmp_path / "events.sqlite3"),
        monitor=FakeMonitor(),
        manager=FakeManager(),
    )
    runtime = FakeExpertRuntime()
    controller.expert_runtime = runtime

    controller.mark_inference_phase("prefill", True, "request-a")
    controller.mark_inference_phase("decode", request_id="request-a")
    controller.mark_inference_phase("prefill", True, "request-b")
    controller.mark_inference_phase("idle", request_id="request-a")
    controller.mark_inference_phase("decode", request_id="request-b")
    controller.mark_inference_phase("idle", request_id="request-b")

    assert runtime.calls == [
        ("prefill", True),
        ("decode", False),
        ("prefill", False),
        ("prefill", False),
        ("decode", False),
        ("idle", False),
    ]
    assert controller._inference_phases == {}


def test_auto_resize_requires_a_guardrail_approved_decode_plan(
    tmp_path: Path,
) -> None:
    controller = PLLMController(
        PLLMConfig(
            model_path=str(tmp_path / "missing-model"),
            hibercache_dir=str(tmp_path / "cache"),
            expert_runtime_socket=str(tmp_path / "pllm-eer.sock"),
            expert_auto_resize_enabled=True,
            expert_resize_cooldown_seconds=0,
        ),
        Storage(tmp_path / "events.sqlite3"),
        monitor=FakeMonitor(),
        manager=FakeManager(),
    )
    calls = []
    controller.expert_dataplane_action = calls.append
    status = {
        "data_plane_ready": True,
        "plan": {"action": "elastic_resident"},
        "decode_plan": {"action": "observe", "slots_per_layer": 512},
        "data_plane": {
            "data_plane": {"layers": [{"slot_count": 512}]}
        },
    }

    assert controller._maybe_auto_resize(status) is False
    assert calls == []

    status["decode_plan"] = {
        "action": "decode_elastic",
        "slots_per_layer": 496,
        "estimated_slowdown_ratio": 1.4,
    }
    assert controller._maybe_auto_resize(status) is True
    assert calls == [
        {
            "action": "resize",
            "slots_per_layer": 496,
            "retain_policy": "decode_hot",
        }
    ]


def test_new_prefill_expands_an_elastic_runtime_before_forwarding(
    tmp_path: Path,
) -> None:
    controller = PLLMController(
        PLLMConfig(
            model_path=str(tmp_path / "missing-model"),
            hibercache_dir=str(tmp_path / "cache"),
            expert_runtime_socket=str(tmp_path / "pllm-eer.sock"),
        ),
        Storage(tmp_path / "events.sqlite3"),
        monitor=FakeMonitor(),
        manager=FakeManager(),
    )

    class ElasticRuntime(FakeExpertRuntime):
        def status(self):
            return {
                "online": True,
                "data_plane_ready": True,
                "slots_per_layer": 496,
                "data_plane": {
                    "layers": [
                        {"slot_count": 496, "global_experts": 512}
                    ]
                },
            }

    runtime = ElasticRuntime()
    controller.expert_runtime = runtime
    actions = []
    controller.expert_dataplane_action = actions.append

    controller.prepare_inference_request("request-new")

    assert runtime.calls == [("prefill", True)]
    assert actions == [
        {
            "action": "resize",
            "slots_per_layer": 512,
            "retain_policy": "lru",
            "phase": "prefill",
        }
    ]
