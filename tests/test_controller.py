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


def test_manual_yield_escalation_and_wake(tmp_path: Path) -> None:
    manager = FakeManager()
    monitor = FakeMonitor()
    storage = Storage(tmp_path / "events.sqlite3")
    controller = PLLMController(
        PLLMConfig(hibercache_dir=str(tmp_path / "cache")),
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
