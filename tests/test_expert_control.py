from __future__ import annotations

import time
from pathlib import Path

from pllm.config import PLLMConfig
from pllm.expert_control import ExpertResidencyControlPlane
from pllm.expert_residency import ResourceEnvelope
from pllm.models import SensorSnapshot, WorkloadClass
from tests.test_expert_catalog import _write_fake_model


def test_control_plane_exposes_catalog_and_never_claims_execution(
    tmp_path: Path,
) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(PLLMConfig(model_path=str(model)))

    status = control.status()
    planned = control.plan(
        ResourceEnvelope(
            total_memory_gib=1,
            foreground_reserve_gib=0,
            system_reserve_gib=0,
        ),
        byte_hit_rate=0.95,
        workload="test",
    )

    assert status["available"] is True
    assert status["model"]["expert_object_count"] == 8
    assert planned["data_plane_ready"] is False
    assert planned["plan"]["executable"] is False
    assert planned["plan"]["exact_route_required"] is True
    assert planned["guardrail"] == "recommendation_only_no_vllm_weight_mutation"


def test_workload_recommendation_is_hypothetical(tmp_path: Path) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(PLLMConfig(model_path=str(model)))

    result = control.recommend(
        SensorSnapshot(timestamp=1, memory_total_gb=128), WorkloadClass.CREATIVE
    )

    assert result["plan"]["workload"] == "creative"
    assert result["plan"]["evidence"] == (
        "hypothetical_control_input_not_model_measurement"
    )
    assert result["plan"]["data_plane_ready"] is False


def test_discrete_gpu_uses_vram_capacity_not_system_ram(tmp_path: Path) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(PLLMConfig(model_path=str(model)))

    result = control.recommend(
        SensorSnapshot(
            timestamp=1,
            memory_total_gb=256,
            gpu_memory_total_gb=96,
            uma=False,
        ),
        WorkloadClass.GPU_PRESSURE,
    )

    assert result["plan"]["capacity_scope"] == "discrete_gpu_vram"
    assert result["plan"]["envelope"]["total_memory_gib"] == 96
    assert result["plan"]["envelope"]["foreground_reserve_gib"] == 8


def test_capacity_generation_changes_only_when_the_envelope_changes(
    tmp_path: Path,
) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(PLLMConfig(model_path=str(model)))
    snapshot = SensorSnapshot(timestamp=1, gpu_memory_total_gb=96, uma=False)

    first = control.recommend(snapshot, WorkloadClass.IDLE)["plan"]["generation"]
    repeated = control.recommend(snapshot, WorkloadClass.IDLE)["plan"]["generation"]
    changed = control.recommend(snapshot, WorkloadClass.CREATIVE)["plan"]["generation"]

    assert repeated == first
    assert changed == first + 1


def test_uma_uses_shared_system_capacity(tmp_path: Path) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(PLLMConfig(model_path=str(model)))

    result = control.recommend(
        SensorSnapshot(
            timestamp=1,
            memory_total_gb=128,
            gpu_memory_total_gb=128,
            uma=True,
        ),
        WorkloadClass.CREATIVE,
    )

    assert result["plan"]["capacity_scope"] == "coherent_uma"
    assert result["plan"]["envelope"]["total_memory_gib"] == 128
    assert result["plan"]["envelope"]["foreground_reserve_gib"] == 64


def test_missing_catalog_reports_unavailable(tmp_path: Path) -> None:
    control = ExpertResidencyControlPlane(
        PLLMConfig(model_path=str(tmp_path / "missing"))
    )
    status = control.status()

    assert status["available"] is False
    assert status["data_plane_ready"] is False
    assert status["error"]


def test_decode_plan_requires_live_decode_observations(tmp_path: Path) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(
        PLLMConfig(
            model_path=str(model),
            decode_candidate_slots=[2, 3],
            decode_target_reclaim_gib=1e-8,
            decode_planner_async=False,
        )
    )

    result = control.plan_decode_residency(
        {
            "route_trace": {
                "phase": "decode",
                "decode_observations": 2,
                "next_window": {"prediction_ready": False},
            },
            "decode_horizon": {"remaining_tokens": 512},
        }
    )

    assert result["action"] == "observe"
    assert result["slots_per_layer"] == 4


def test_async_decode_planner_returns_a_fast_pending_yield(tmp_path: Path) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(
        PLLMConfig(
            model_path=str(model),
            decode_candidate_slots=[2, 3],
            decode_target_reclaim_gib=1e-8,
            decode_planner_async=True,
        )
    )
    runtime = {
        "route_trace": {
            "phase": "decode",
            "decode_observations": 2,
            "next_window": {"prediction_ready": False},
        },
        "decode_horizon": {"remaining_tokens": 512},
    }

    started = time.perf_counter()
    first = control.plan_decode_residency(runtime)
    elapsed = time.perf_counter() - started
    for _ in range(100):
        completed = control.plan_decode_residency(runtime)
        if not completed.get("planner_pending"):
            break
        time.sleep(0.001)

    assert elapsed < 0.1
    assert first["action"] == "yield"
    assert first["planner_pending"] is True
    assert completed["action"] == "observe"


def test_async_decode_planner_never_returns_a_stale_route_generation(
    tmp_path: Path,
) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(
        PLLMConfig(model_path=str(model), decode_planner_async=True)
    )

    def delayed(runtime):
        time.sleep(0.01)
        generation = runtime["route_trace"]["next_window"]["route_generation"]
        return {"action": "observe", "route_generation": generation}

    control._plan_decode_residency_sync = delayed
    first = {
        "route_trace": {
            "phase": "decode",
            "next_window": {
                "request_generation": 1,
                "route_generation": 10,
            },
        },
        "decode_horizon": {"remaining_tokens": 512, "decode_requests": 1},
    }
    second = {
        **first,
        "route_trace": {
            "phase": "decode",
            "next_window": {
                "request_generation": 2,
                "route_generation": 20,
            },
        },
    }
    assert control.plan_decode_residency(first)["planner_pending"] is True
    assert control.plan_decode_residency(second)["planner_pending"] is True
    for _ in range(200):
        result = control.plan_decode_residency(second)
        if not result.get("planner_pending"):
            break
        time.sleep(0.001)

    assert result["route_generation"] == 20


def test_decode_plan_uses_guardrail_after_window_is_warm(tmp_path: Path) -> None:
    model = _write_fake_model(tmp_path / "model")
    control = ExpertResidencyControlPlane(
        PLLMConfig(
            model_path=str(model),
            decode_candidate_slots=[2, 3],
            decode_target_reclaim_gib=1e-8,
            decode_min_byte_hit_rate=0.95,
            decode_baseline_tpot_ms=100,
            expert_requested_token_rate=1,
            expert_io_budget_gib_s=1,
            decode_planner_async=False,
        )
    )

    result = control.plan_decode_residency(
        {
            "route_trace": {
                "phase": "decode",
                "decode_observations": 8,
                "next_window": {
                    "prediction_ready": True,
                    "minimum_completed_windows": 2,
                    "profiles": {
                        "2": {"per_layer": {}},
                        "3": {
                            "per_layer": {
                                str(layer): {
                                    "byte_hit_rate_lower_bound": 0.99,
                                    "mean_misses_per_token_upper_bound": 0.02,
                                    "p95_misses_per_token_upper_bound": 1,
                                    "heldout_windows": 1,
                                }
                                for layer in (0, 1)
                            }
                        },
                    },
                },
            },
            "decode_horizon": {"remaining_tokens": 512},
            "data_plane": {
                "layers": [
                    {"layer": 0, "slot_count": 4},
                    {"layer": 1, "slot_count": 4},
                ]
            },
        }
    )

    assert result["action"] == "decode_elastic"
    assert result["slots_per_layer"] == 3
    assert result["latency_guardrail"] == "heldout_next_window_strictly_below_5x"
