from __future__ import annotations

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


def test_missing_catalog_reports_unavailable(tmp_path: Path) -> None:
    control = ExpertResidencyControlPlane(
        PLLMConfig(model_path=str(tmp_path / "missing"))
    )
    status = control.status()

    assert status["available"] is False
    assert status["data_plane_ready"] is False
    assert status["error"]
