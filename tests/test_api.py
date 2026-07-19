from __future__ import annotations

from pathlib import Path

from pllm.api import create_app
from pllm.storage import Storage


class FakeController:
    def __init__(self, target: str, active: bool = True) -> None:
        self.target = target
        self.active = active
        self.action_calls = []
        self.phase_calls = []

    def status(self):
        return {
            "state": "active" if self.active else "hot_sleep",
            "mode": "auto",
            "services": [],
        }

    def services(self):
        return []

    def refresh_services(self):
        return []

    def capabilities(self, refresh=False):
        return {"vllm": {"version": "0.25.1"}, "refresh": refresh}

    def expert_residency_status(self):
        return {
            "available": True,
            "data_plane_ready": False,
            "backend": "control_plane_only",
            "plan": {},
        }

    def plan_expert_residency(self, payload):
        return {
            "available": True,
            "data_plane_ready": False,
            "backend": "control_plane_only",
            "plan": {
                "action": "elastic_resident",
                "slots_per_layer": 128,
                "evidence": "manual_control_input_not_model_measurement",
            },
            "input": payload,
        }

    def expert_dataplane_status(self):
        return {
            "online": True,
            "data_plane_ready": True,
            "backend": "test_exact_slot_backend",
        }

    def expert_dataplane_action(self, payload):
        self.action_calls.append(("expert_dataplane", payload))
        return {"ok": True, "input": payload}

    def compile_policy(self, text, apply=False):
        return {"input": text, "applied": apply, "rules": []}

    def update_policy(self, values):
        return values

    def action(self, action, level=None):
        self.action_calls.append((action, level))
        return self.status()

    def can_proxy(self):
        return self.active

    def proxy_target(self):
        return self.target

    def mark_inference_phase(self, phase, reset_decode=False, request_id=""):
        self.phase_calls.append((phase, reset_decode, request_id))

    def prepare_inference_request(self, request_id):
        self.mark_inference_phase("prefill", True, request_id)


def test_proxy_records_completed_request(mock_vllm_url: str, tmp_path: Path) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    controller = FakeController(mock_vllm_url)
    app = create_app(controller, storage)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    replay_id = response.headers["X-PLLM-Replay-ID"]
    replay = storage.get_replay(replay_id)
    assert replay is not None
    assert replay["status"] == "completed"
    assert replay["response_text"] == "Mock response: hello"
    assert [(phase, reset) for phase, reset, _ in controller.phase_calls] == [
        ("prefill", True),
        ("idle", False),
    ]
    assert {request_id for _, _, request_id in controller.phase_calls} == {
        replay_id
    }


def test_paused_proxy_queues_replay(mock_vllm_url: str, tmp_path: Path) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    app = create_app(FakeController(mock_vllm_url, active=False), storage)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "later"}]},
    )

    assert response.status_code == 503
    replay = storage.get_replay(response.headers["X-PLLM-Replay-ID"])
    assert replay is not None
    assert replay["status"] == "queued"


def test_control_action_endpoint(mock_vllm_url: str, tmp_path: Path) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    controller = FakeController(mock_vllm_url)
    app = create_app(controller, storage)
    client = app.test_client()

    response = client.post("/api/v1/actions", json={"action": "pause", "level": 2})

    assert response.status_code == 200
    assert controller.action_calls == [("pause", 2)]


def test_dashboard_and_extended_control_endpoints(
    mock_vllm_url: str, tmp_path: Path
) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    app = create_app(FakeController(mock_vllm_url), storage)
    client = app.test_client()

    dashboard = client.get("/")
    capabilities = client.get("/api/v1/capabilities?refresh=1")
    compiled = client.post(
        "/api/v1/policy/compile", json={"text": "Blender 时释放", "apply": False}
    )

    assert dashboard.status_code == 200
    assert b"PLLM HiberFlow-EER" in dashboard.data
    assert capabilities.json["refresh"] is True
    assert compiled.json["input"] == "Blender 时释放"


def test_expert_residency_endpoints_are_explicitly_non_executable(
    mock_vllm_url: str, tmp_path: Path
) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    app = create_app(FakeController(mock_vllm_url), storage)
    client = app.test_client()

    status = client.get("/api/v1/expert-residency")
    plan = client.post(
        "/api/v1/expert-residency/plan",
        json={
            "workload": "creative",
            "byte_hit_rate": 0.95,
            "envelope": {"foreground_reserve_gib": 64},
        },
    )

    assert status.status_code == 200
    assert status.json["data_plane_ready"] is False
    assert plan.status_code == 200
    assert plan.json["data_plane_ready"] is False
    assert plan.json["plan"]["evidence"] == (
        "manual_control_input_not_model_measurement"
    )


def test_expert_dataplane_actions_are_separate_from_projection(
    mock_vllm_url: str, tmp_path: Path
) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    controller = FakeController(mock_vllm_url)
    app = create_app(controller, storage)
    client = app.test_client()

    status = client.get("/api/v1/expert-dataplane")
    resize = client.post(
        "/api/v1/expert-dataplane/actions",
        json={"action": "resize", "slots_per_layer": 128},
    )

    assert status.json["data_plane_ready"] is True
    assert resize.status_code == 200
    assert controller.action_calls[-1][0] == "expert_dataplane"


def test_stream_backend_failure_returns_502(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    app = create_app(FakeController("http://127.0.0.1:1"), storage)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "missing",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.headers.get("X-PLLM-Replay-ID")


def test_stream_persists_incremental_output(
    mock_vllm_url: str, tmp_path: Path
) -> None:
    storage = Storage(tmp_path / "events.sqlite3")
    controller = FakeController(mock_vllm_url)
    app = create_app(controller, storage)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "stream": True,
            "messages": [{"role": "user", "content": "stream me"}],
        },
        buffered=True,
    )

    replay = storage.get_replay(response.headers["X-PLLM-Replay-ID"])
    assert response.status_code == 200
    assert replay is not None
    assert replay["status"] == "completed"
    assert replay["generated_tokens"] > 0
    assert replay["response_text"] == "Mock response: stream me "
    assert [(phase, reset) for phase, reset, _ in controller.phase_calls] == [
        ("prefill", True),
        ("decode", False),
        ("idle", False),
    ]
