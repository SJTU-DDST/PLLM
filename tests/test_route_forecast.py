from __future__ import annotations

import pytest

from pllm.route_forecast import (
    RouteMTPResidencyPredictor,
    SequentialCoverageCalibrator,
)


def test_coverage_gate_stays_closed_without_enough_heldout_samples() -> None:
    calibrator = SequentialCoverageCalibrator(
        [3, 4],
        target_miss_rate=0.2,
        confidence_delta=0.9,
        minimum_samples=3,
    )
    calibrator.observe({3: False, 4: False})
    calibrator.observe({3: False, 4: False})

    assert calibrator.estimate(3).certified is False
    assert calibrator.estimate(3).samples == 2


def test_repeated_routes_eventually_certify_a_safe_profile() -> None:
    predictor = RouteMTPResidencyPredictor(
        [0, 1],
        4,
        1,
        candidate_slots=[3, 4],
        target_miss_rate=0.2,
        confidence_delta=0.9,
        minimum_calibration_samples=3,
    )
    for _ in range(12):
        predictor.observe_step({0: [0], 1: [0]})

    plan = predictor.residency_plan(3)
    assert plan["action"] == "prefetch_and_evict"
    assert plan["coverage"]["misses"] == 0
    assert plan["exact_route_authoritative"] is True
    assert plan["exact_miss_fallback_required"] is True


def test_any_layer_miss_fails_the_step_level_coverage_event() -> None:
    predictor = RouteMTPResidencyPredictor(
        [0, 1],
        4,
        1,
        candidate_slots=[2, 4],
        minimum_calibration_samples=1,
    )
    predictor.observe_step({0: [0], 1: [0]})
    outcome = predictor.observe_step({0: [0], 1: [3]})

    assert outcome["missed_by_slots"]["2"] is True
    assert predictor.calibrator.estimate(2).misses == 1
    assert predictor.calibrator.estimate(4).misses == 0


def test_direct_route_head_scores_drive_the_forecast_without_changing_exactness() -> None:
    predictor = RouteMTPResidencyPredictor(
        [7],
        8,
        2,
        candidate_slots=[3, 8],
    )
    scores = [0.0] * 8
    scores[6] = 10.0
    scores[7] = 9.0
    forecast = predictor.forecast(3, direct_scores={7: scores})[7]

    assert {6, 7}.issubset(forecast.resident_experts)
    assert forecast.mtp_signal_used is True
    assert predictor.residency_plan(3, direct_scores={7: scores})["action"] == "shadow_only"


def test_mtp_cross_router_signal_learns_target_layer_associations() -> None:
    predictor = RouteMTPResidencyPredictor(
        [0],
        8,
        1,
        candidate_slots=[2, 8],
    )
    for _ in range(4):
        predictor.observe_step({0: [6]}, mtp_experts=[3])

    forecast = predictor.forecast(2, mtp_experts=[3])[0]
    assert 6 in forecast.resident_experts
    assert forecast.mtp_signal_used is True
    assert predictor.status()["mtp_signal_attached"] is True


def test_predictor_rejects_partial_or_out_of_range_exact_routes() -> None:
    predictor = RouteMTPResidencyPredictor([0, 1], 4, 1, candidate_slots=[2, 4])

    with pytest.raises(ValueError, match="layer mismatch"):
        predictor.observe_step({0: [0]})
    with pytest.raises(ValueError, match="out-of-range"):
        predictor.observe_step({0: [0], 1: [4]})
