from __future__ import annotations

from pathlib import Path

import pytest

from pllm.expert_dataplane import ExpertSlotDataPlane
from pllm.expert_store import (
    ExpertPackageCodec,
    ExpertPayload,
    SSDExpertStore,
)


FINGERPRINT = "f" * 64
FORMAT = "vllm_runtime_nvfp4_marlin_v1"


class MemorySource:
    def __init__(self, experts: int = 8) -> None:
        self.payloads = {
            (0, expert): ExpertPayload.create(
                0,
                expert,
                FORMAT,
                FINGERPRINT,
                [("w13_weight", "uint8", (4,), bytes([expert]) * 4)],
            )
            for expert in range(experts)
        }

    def contains(self, layer: int, expert: int) -> bool:
        return (layer, expert) in self.payloads

    def get(self, layer: int, expert: int) -> ExpertPayload:
        return self.payloads[(layer, expert)]


class MemorySink:
    required_format = FORMAT

    def __init__(self, slots: int) -> None:
        self.slot_count = slots
        self.rows: list[bytes | None] = [None] * slots
        self.mapping: dict[int, int] = {}

    def write(self, slot: int, payload: ExpertPayload) -> None:
        self.rows[slot] = payload.tensor_bytes("w13_weight")

    def invalidate(self, slot: int) -> None:
        del slot

    def publish_mapping(self, mapping: dict[int, int], generation: int) -> None:
        del generation
        self.mapping = dict(mapping)

    def begin_resize(self, slot_count: int) -> None:
        self.slot_count = slot_count
        self.rows = [None] * slot_count

    def finish_resize(self) -> None:
        pass


def test_package_checksum_and_atomic_ssd_round_trip(tmp_path: Path) -> None:
    payload = MemorySource().get(0, 3)
    encoded = ExpertPackageCodec.encode(payload)
    assert ExpertPackageCodec.decode(encoded) == payload

    store = SSDExpertStore(tmp_path, FINGERPRINT, required_format=FORMAT)
    path = store.put(payload)
    assert path.is_file()
    assert store.get(0, 3) == payload

    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum"):
        store.get(0, 3)


def test_actual_route_miss_loads_exact_expert_before_publish() -> None:
    sink = MemorySink(2)
    plane = ExpertSlotDataPlane(MemorySource())
    plane.register_layer(0, 8, sink, initial_experts=[0, 1])

    mapping = plane.ensure(0, [1, 5])

    assert set(mapping) == {1, 5}
    assert sink.rows[mapping[5]] == bytes([5]) * 4
    assert sink.mapping[5] == mapping[5]
    assert 0 not in sink.mapping


def test_resize_reloads_retained_experts_from_backing_store() -> None:
    sink = MemorySink(4)
    plane = ExpertSlotDataPlane(MemorySource())
    plane.register_layer(0, 8, sink, initial_experts=[0, 1, 2, 3])

    status = plane.resize(0, 2, retain=[2, 3])

    assert status["slot_count"] == 2
    assert set(status["logical_to_slot"]) == {2, 3}
    assert set(sink.mapping) == {2, 3}
