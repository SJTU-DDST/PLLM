from __future__ import annotations

import os
from pathlib import Path

from pllm.config import PLLMConfig, pllm_runtime_dir
from pllm.storage import Storage
from scripts.calibrate_agent import percentile


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    config = PLLMConfig(
        api_port=19000,
        game_patterns=["steam", "test-game"],
        expert_io_budget_gib_s=1.25,
    )
    config.save(path)

    loaded = PLLMConfig.load(path)

    assert loaded.api_port == 19000
    assert loaded.game_patterns == ["steam", "test-game"]
    assert loaded.default_vllm_urls == ["http://127.0.0.1:8000"]
    assert loaded.expert_io_budget_gib_s == 1.25
    assert loaded.expert_data_plane_enabled is True
    assert loaded.expert_auto_resize_enabled is False


def test_runtime_dir_uses_system_runtime_when_configured_directory_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing"
    system_runtime = Path(f"/run/user/{os.getuid()}")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(missing))
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: path == system_runtime,
    )
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: Path(path) == system_runtime,
    )

    assert pllm_runtime_dir() == system_runtime


def test_runtime_dir_falls_back_to_tmp_when_runtime_directories_are_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(Path, "is_dir", lambda path: False)

    assert pllm_runtime_dir() == Path("/tmp") / f"pllm-{os.getuid()}"


def test_storage_records_events_and_replays(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    event_id = storage.add_event("sleep", "hot_sleep", "game", {"level": 1})
    replay_id = storage.create_replay({"messages": [{"content": "hello"}]}, "queued")
    storage.update_replay(replay_id, "completed", response_text="world")

    assert event_id > 0
    assert storage.list_events()[0]["payload"]["level"] == 1
    replay = storage.get_replay(replay_id)
    assert replay is not None
    assert replay["request"]["messages"][0]["content"] == "hello"
    assert replay["response_text"] == "world"


def test_storage_preserves_token_boundary_and_experiment(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    replay_id = storage.create_replay({"messages": [{"content": "code"}]}, "running")
    storage.update_replay_progress(replay_id, 197, "partial output")

    assert storage.pause_running_replays() == 1
    replay = storage.get_replay(replay_id)
    assert replay is not None
    assert replay["status"] == "paused"
    assert replay["paused_at_token"] == 197
    assert replay["response_text"] == "partial output"
    assert storage.resume_paused_replays() == 1

    experiment_id = storage.add_experiment("safe_probe", "host only", {"gpu": False})
    experiment = storage.list_experiments()[0]
    assert experiment["id"] == experiment_id
    assert experiment["metrics"]["gpu"] is False


def test_calibration_uses_observed_p95() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0]) == 40.0
