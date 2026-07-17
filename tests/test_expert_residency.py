from __future__ import annotations

from pathlib import Path

from pllm.expert_catalog import ExpertCatalog
from pllm.expert_residency import (
    ConformalExpertPredictor,
    ExpertCacheSimulator,
    ResidencyPlanner,
    ResourceEnvelope,
    RouteHistoryPredictor,
)
from pllm.expert_trace import synthetic_trace
from tests.test_expert_catalog import _write_fake_model


def _trained_predictor(catalog: ExpertCatalog):
    records = list(synthetic_trace(catalog, requests=3, tokens_per_request=12))
    train = [record for record in records if record.request_id == "synthetic-0"]
    calibration = [
        record for record in records if record.request_id == "synthetic-1"
    ]
    test = [record for record in records if record.request_id == "synthetic-2"]
    base = RouteHistoryPredictor(catalog.experts_per_layer)
    base.fit(train)
    predictor = ConformalExpertPredictor(base, alpha=0.1)
    report = predictor.calibrate(calibration)
    return predictor, report, test


def test_conformal_prediction_set_reports_marginal_scope(tmp_path: Path) -> None:
    catalog = ExpertCatalog.from_model(_write_fake_model(tmp_path / "model"))
    predictor, report, test = _trained_predictor(catalog)
    evaluation = predictor.evaluate(test)

    assert report.records == 24
    assert report.guarantee_scope == "split_conformal_marginal_under_exchangeability"
    assert 0 <= evaluation["coverage"] <= 1
    assert all(
        catalog.active_experts_per_token
        <= len(predictor.prediction_set(layer))
        <= catalog.experts_per_layer
        for layer in catalog.moe_layers
    )


def test_cache_simulator_preserves_actual_route_and_accounts_bytes(
    tmp_path: Path,
) -> None:
    catalog = ExpertCatalog.from_model(_write_fake_model(tmp_path / "model"))
    predictor, _report, test = _trained_predictor(catalog)

    result = ExpertCacheSimulator(catalog, predictor, slots_per_layer=2).run(test)

    assert result.exact_route_preserved is True
    assert result.actual_bytes > 0
    assert result.actual_bytes >= result.resident_hit_bytes
    assert result.blocking_miss_bytes >= 0
    assert result.evidence == "synthetic_no_gpu"


def test_phase_boundary_selects_full_elastic_yield_and_hibernate(
    tmp_path: Path,
) -> None:
    catalog = ExpertCatalog.from_model(_write_fake_model(tmp_path / "model"))
    planner = ResidencyPlanner(catalog, slot_profiles=(2, 3, 4))

    full = planner.plan(
        ResourceEnvelope(
            total_memory_gib=1,
            foreground_reserve_gib=0,
            system_reserve_gib=0,
            compute_duty_cycle=1,
        ),
        byte_hit_rate=1,
    )
    elastic = planner.plan(
        ResourceEnvelope(
            total_memory_gib=(
                catalog.non_routed_bytes + catalog.routed_expert_bytes * 3 / 4
            )
            / 1024**3
            + 1e-12,
            foreground_reserve_gib=0,
            system_reserve_gib=0,
            compute_duty_cycle=0.5,
            io_budget_gib_s=10,
        ),
        byte_hit_rate=0.9,
    )
    yielded = planner.plan(
        ResourceEnvelope(
            total_memory_gib=1,
            foreground_reserve_gib=0,
            system_reserve_gib=0,
            compute_duty_cycle=0,
        ),
        byte_hit_rate=1,
    )
    hibernated = planner.plan(
        ResourceEnvelope(
            total_memory_gib=0,
            foreground_reserve_gib=0,
            system_reserve_gib=0,
        ),
        byte_hit_rate=0,
    )

    assert full.action == "full_resident"
    assert elastic.action == "elastic_resident"
    assert elastic.exact_route_required is True
    assert yielded.action == "yield"
    assert hibernated.action == "hibernate"
    assert all(
        plan.data_plane_ready is False
        for plan in (full, elastic, yielded, hibernated)
    )
