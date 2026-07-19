from __future__ import annotations

import pytest

from pllm.pause_resume import (
    PauseResumeInputs,
    PauseResumePlanner,
    ResumeStrategy,
    SparseBlockLayout,
    pack_active_blocks,
    restore_active_blocks,
)


def test_sparse_block_component_round_trip_preserves_only_selected_blocks() -> None:
    source = bytes(range(32))
    layout = SparseBlockLayout(total_blocks=8, block_bytes=4, active_blocks=(1, 4, 7))

    component = pack_active_blocks("attention-kv", source, layout, dtype="fp8")
    destination = bytearray(b"\xff" * len(source))
    restored_layout = restore_active_blocks(component, destination)

    assert restored_layout == layout
    assert len(component.data) == 12
    for block in range(8):
        start = block * 4
        expected = source[start : start + 4] if block in layout.active_blocks else b"\xff" * 4
        assert destination[start : start + 4] == expected


def test_sparse_block_layout_rejects_ambiguous_indexes() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        SparseBlockLayout(total_blocks=4, block_bytes=8, active_blocks=(2, 1))
    with pytest.raises(ValueError, match="outside"):
        SparseBlockLayout(total_blocks=4, block_bytes=8, active_blocks=(4,))


def test_planner_prefers_retained_cpu_for_live_sparse_state() -> None:
    gib = 1024**3
    profile = PauseResumeInputs(
        live_tokens=32_768,
        capacity_tokens=1_200_000,
        block_tokens=16,
        allocated_kv_bytes=8 * gib,
        cpu_staging_bytes=512 * 1024**2,
        ssd_read_bytes_per_second=3 * gib,
        ssd_write_bytes_per_second=2 * gib,
        host_to_device_bytes_per_second=12 * gib,
        device_to_host_bytes_per_second=12 * gib,
        prefill_tokens_per_second=500,
        cpu_primary_quiesce_supported=True,
    )

    planner = PauseResumePlanner(profile)
    estimates = {item.strategy: item for item in planner.estimates()}

    assert estimates[ResumeStrategy.KEEP_GPU].feasible is False
    assert estimates[ResumeStrategy.ACTIVE_CPU].state_bytes < profile.cpu_staging_bytes
    assert planner.choose().strategy is ResumeStrategy.ACTIVE_CPU
    assert (
        estimates[ResumeStrategy.ACTIVE_SSD].wake_seconds
        < estimates[ResumeStrategy.FULL_SSD].wake_seconds
    )


def test_planner_falls_back_when_cpu_tier_is_too_small() -> None:
    mib = 1024**2
    profile = PauseResumeInputs(
        live_tokens=16_384,
        capacity_tokens=32_768,
        block_tokens=16,
        allocated_kv_bytes=1024 * mib,
        cpu_staging_bytes=8 * mib,
        ssd_read_bytes_per_second=1000 * mib,
        ssd_write_bytes_per_second=1000 * mib,
        host_to_device_bytes_per_second=10_000 * mib,
        device_to_host_bytes_per_second=10_000 * mib,
        prefill_tokens_per_second=10,
    )

    estimates = {item.strategy: item for item in PauseResumePlanner(profile).estimates()}

    assert estimates[ResumeStrategy.ACTIVE_CPU].feasible is False
    assert PauseResumePlanner(profile).choose().strategy is ResumeStrategy.ACTIVE_SSD


def test_planner_requires_explicit_cpu_primary_quiesce_capability() -> None:
    mib = 1024**2
    profile = PauseResumeInputs(
        live_tokens=1024,
        capacity_tokens=32_768,
        block_tokens=16,
        allocated_kv_bytes=1024 * mib,
        cpu_staging_bytes=512 * mib,
        ssd_read_bytes_per_second=1000 * mib,
        ssd_write_bytes_per_second=1000 * mib,
        host_to_device_bytes_per_second=10_000 * mib,
        device_to_host_bytes_per_second=10_000 * mib,
        prefill_tokens_per_second=10,
    )

    estimates = {item.strategy: item for item in PauseResumePlanner(profile).estimates()}

    assert estimates[ResumeStrategy.ACTIVE_CPU].feasible is False
    assert "cannot quiesce" in estimates[ResumeStrategy.ACTIVE_CPU].reason
    assert PauseResumePlanner(profile).choose().strategy is ResumeStrategy.ACTIVE_SSD
