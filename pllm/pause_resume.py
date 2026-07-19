from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .hiberstate import StateComponent


SPARSE_BLOCK_FORMAT = "pllm-active-blocks-v1"


class ResumeStrategy(StrEnum):
    KEEP_GPU = "keep_gpu"
    RECOMPUTE = "recompute"
    ACTIVE_CPU = "active_blocks_cpu"
    ACTIVE_SSD = "active_blocks_ssd"
    FULL_SSD = "full_kv_ssd"


@dataclass(slots=True, frozen=True)
class SparseBlockLayout:
    """An exact, lossless selection of logical blocks from a dense buffer."""

    total_blocks: int
    block_bytes: int
    active_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.total_blocks <= 0 or self.block_bytes <= 0:
            raise ValueError("total_blocks and block_bytes must be positive")
        normalized = tuple(sorted(set(self.active_blocks)))
        if normalized != self.active_blocks:
            raise ValueError("active_blocks must be sorted and unique")
        if not normalized:
            raise ValueError("active_blocks cannot be empty")
        if normalized[0] < 0 or normalized[-1] >= self.total_blocks:
            raise ValueError("active block index is outside the logical buffer")

    @property
    def logical_size_bytes(self) -> int:
        return self.total_blocks * self.block_bytes

    @property
    def packed_size_bytes(self) -> int:
        return len(self.active_blocks) * self.block_bytes

    def pack(self, source: bytes | bytearray | memoryview) -> bytes:
        view = memoryview(source).cast("B")
        if len(view) != self.logical_size_bytes:
            raise ValueError("source size does not match sparse block layout")
        packed = bytearray(self.packed_size_bytes)
        destination = memoryview(packed)
        for packed_index, block_index in enumerate(self.active_blocks):
            source_start = block_index * self.block_bytes
            packed_start = packed_index * self.block_bytes
            destination[packed_start : packed_start + self.block_bytes] = view[
                source_start : source_start + self.block_bytes
            ]
        return bytes(packed)

    def restore_into(
        self,
        packed: bytes | bytearray | memoryview,
        destination: bytearray | memoryview,
    ) -> None:
        source = memoryview(packed).cast("B")
        target = memoryview(destination).cast("B")
        if len(source) != self.packed_size_bytes:
            raise ValueError("packed size does not match sparse block layout")
        if len(target) != self.logical_size_bytes:
            raise ValueError("destination size does not match sparse block layout")
        for packed_index, block_index in enumerate(self.active_blocks):
            source_start = packed_index * self.block_bytes
            target_start = block_index * self.block_bytes
            target[target_start : target_start + self.block_bytes] = source[
                source_start : source_start + self.block_bytes
            ]

    def metadata(self) -> dict[str, Any]:
        return {
            "storage_format": SPARSE_BLOCK_FORMAT,
            "total_blocks": self.total_blocks,
            "block_bytes": self.block_bytes,
            "active_blocks": list(self.active_blocks),
            "logical_size_bytes": self.logical_size_bytes,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> SparseBlockLayout:
        if metadata.get("storage_format") != SPARSE_BLOCK_FORMAT:
            raise ValueError("component is not an active-block snapshot")
        return cls(
            total_blocks=int(metadata["total_blocks"]),
            block_bytes=int(metadata["block_bytes"]),
            active_blocks=tuple(int(item) for item in metadata["active_blocks"]),
        )


def pack_active_blocks(
    name: str,
    source: bytes | bytearray | memoryview,
    layout: SparseBlockLayout,
    *,
    dtype: str = "bytes",
    shape: tuple[int, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> StateComponent:
    component_metadata = dict(metadata or {})
    component_metadata.update(layout.metadata())
    return StateComponent(
        name=name,
        data=layout.pack(source),
        dtype=dtype,
        shape=shape,
        metadata=component_metadata,
    )


def restore_active_blocks(
    component: StateComponent, destination: bytearray | memoryview
) -> SparseBlockLayout:
    layout = SparseBlockLayout.from_metadata(component.metadata)
    layout.restore_into(component.data, destination)
    return layout


@dataclass(slots=True, frozen=True)
class PauseResumeInputs:
    live_tokens: int
    capacity_tokens: int
    block_tokens: int
    allocated_kv_bytes: int
    cpu_staging_bytes: int
    ssd_read_bytes_per_second: float
    ssd_write_bytes_per_second: float
    host_to_device_bytes_per_second: float
    device_to_host_bytes_per_second: float
    prefill_tokens_per_second: float
    require_gpu_reclaim: bool = True
    cpu_primary_quiesce_supported: bool = False
    recurrent_state_bytes: int = 0
    moe_hotset_bytes: int = 0
    moe_read_bytes_per_second: float = 0.0
    overlap_moe_restore: bool = False

    def __post_init__(self) -> None:
        integer_values = (
            self.live_tokens,
            self.capacity_tokens,
            self.block_tokens,
            self.allocated_kv_bytes,
            self.cpu_staging_bytes,
            self.recurrent_state_bytes,
            self.moe_hotset_bytes,
        )
        if any(value < 0 for value in integer_values):
            raise ValueError("pause/resume sizes cannot be negative")
        if self.capacity_tokens <= 0 or self.block_tokens <= 0:
            raise ValueError("capacity_tokens and block_tokens must be positive")
        if self.live_tokens > self.capacity_tokens:
            raise ValueError("live_tokens cannot exceed capacity_tokens")
        rates = (
            self.ssd_read_bytes_per_second,
            self.ssd_write_bytes_per_second,
            self.host_to_device_bytes_per_second,
            self.device_to_host_bytes_per_second,
            self.prefill_tokens_per_second,
        )
        if any(value <= 0 for value in rates):
            raise ValueError("pause/resume throughput values must be positive")
        if self.moe_hotset_bytes and self.moe_read_bytes_per_second <= 0:
            raise ValueError("MoE read throughput is required for a non-empty hot set")

    @property
    def total_blocks(self) -> int:
        return math.ceil(self.capacity_tokens / self.block_tokens)

    @property
    def active_blocks(self) -> int:
        return math.ceil(self.live_tokens / self.block_tokens) if self.live_tokens else 0

    @property
    def bytes_per_block(self) -> int:
        return math.ceil(self.allocated_kv_bytes / self.total_blocks)

    @property
    def active_state_bytes(self) -> int:
        kv_bytes = min(
            self.allocated_kv_bytes, self.active_blocks * self.bytes_per_block
        )
        return kv_bytes + self.recurrent_state_bytes


@dataclass(slots=True, frozen=True)
class StrategyEstimate:
    strategy: ResumeStrategy
    feasible: bool
    entry_seconds: float
    wake_seconds: float
    state_bytes: int
    reclaimed_kv_bytes: int
    reason: str

    @property
    def interruption_seconds(self) -> float:
        return self.entry_seconds + self.wake_seconds


class PauseResumePlanner:
    """Select an exact resume path from measured local throughput values."""

    def __init__(self, inputs: PauseResumeInputs) -> None:
        self.inputs = inputs

    def estimates(self) -> tuple[StrategyEstimate, ...]:
        profile = self.inputs
        active = profile.active_state_bytes
        full = profile.allocated_kv_bytes + profile.recurrent_state_bytes
        moe_seconds = (
            profile.moe_hotset_bytes / profile.moe_read_bytes_per_second
            if profile.moe_hotset_bytes
            else 0.0
        )

        def with_moe(seconds: float) -> float:
            if not moe_seconds:
                return seconds
            if profile.overlap_moe_restore:
                return max(seconds, moe_seconds)
            return seconds + moe_seconds

        keep_feasible = not profile.require_gpu_reclaim
        active_cpu_feasible = (
            profile.cpu_primary_quiesce_supported
            and active <= profile.cpu_staging_bytes
        )
        return (
            StrategyEstimate(
                ResumeStrategy.KEEP_GPU,
                keep_feasible,
                0.0,
                0.0,
                profile.allocated_kv_bytes,
                0,
                "GPU memory must be reclaimed" if not keep_feasible else "KV remains resident",
            ),
            StrategyEstimate(
                ResumeStrategy.RECOMPUTE,
                True,
                0.0,
                with_moe(profile.live_tokens / profile.prefill_tokens_per_second),
                0,
                profile.allocated_kv_bytes,
                "replay the committed token ledger",
            ),
            StrategyEstimate(
                ResumeStrategy.ACTIVE_CPU,
                active_cpu_feasible,
                active / profile.device_to_host_bytes_per_second,
                with_moe(active / profile.host_to_device_bytes_per_second),
                active,
                profile.allocated_kv_bytes,
                (
                    "active state fits the retained CPU tier"
                    if active_cpu_feasible
                    else (
                        "connector cannot quiesce while retaining its CPU primary"
                        if not profile.cpu_primary_quiesce_supported
                        else "active state exceeds the CPU staging budget"
                    )
                ),
            ),
            StrategyEstimate(
                ResumeStrategy.ACTIVE_SSD,
                True,
                active / profile.device_to_host_bytes_per_second
                + active / profile.ssd_write_bytes_per_second,
                with_moe(
                    active / profile.ssd_read_bytes_per_second
                    + active / profile.host_to_device_bytes_per_second
                ),
                active,
                profile.allocated_kv_bytes,
                "persist only blocks owned by live requests",
            ),
            StrategyEstimate(
                ResumeStrategy.FULL_SSD,
                True,
                full / profile.device_to_host_bytes_per_second
                + full / profile.ssd_write_bytes_per_second,
                with_moe(
                    full / profile.ssd_read_bytes_per_second
                    + full / profile.host_to_device_bytes_per_second
                ),
                full,
                profile.allocated_kv_bytes,
                "baseline full cache image",
            ),
        )

    def choose(self) -> StrategyEstimate:
        feasible = [item for item in self.estimates() if item.feasible]
        if not feasible:
            raise RuntimeError("no feasible pause/resume strategy")
        return min(feasible, key=lambda item: item.interruption_seconds)
