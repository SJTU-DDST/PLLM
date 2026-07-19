from __future__ import annotations

import pytest

from pllm.host_moe_resume import plan_host_moe_resume, rank_recent_experts


def test_recent_rank_uses_frequency_then_recency() -> None:
    ranking = rank_recent_experts([(1, 2), (2, 3), (1, 3)], 6)

    assert ranking[:4] == (1, 3, 2, 0)


def test_host_resume_adds_exact_route_misses_before_execution() -> None:
    history = [
        [(0, 1), (2, 3)],
        [(0, 2), (2, 4)],
    ]
    plan = plan_host_moe_resume(
        history,
        [(5, 0), (2, 5)],
        physical_slots=5,
        hot_slots=3,
        experts_per_layer=6,
    )

    assert plan.exact_misses_by_layer == ((5,), (5,))
    assert plan.naive_copy_objects == 10
    assert plan.critical_copy_objects == 8
    assert plan.expert_copy_reduction_ratio == pytest.approx(0.2)
    assert plan.exact_route_covered is True


def test_host_resume_validates_dimensions_and_capacity() -> None:
    with pytest.raises(ValueError, match="resume slots"):
        plan_host_moe_resume(
            [], [(0,)], physical_slots=2, hot_slots=3, experts_per_layer=4
        )
    with pytest.raises(ValueError, match="every history token"):
        plan_host_moe_resume(
            [[(0,)]],
            [(0,), (1,)],
            physical_slots=2,
            hot_slots=1,
            experts_per_layer=4,
        )
