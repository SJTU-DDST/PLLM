from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from pllm.expert_dataplane import ExpertSlotDataPlane
from pllm.expert_store import (
    ExpertPackageCodec,
    ExpertPayload,
    RDMAPoolExpertStore,
    RDMAPoolStream,
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


class FastMemorySink(MemorySink):
    def resize_with_retained(
        self, slot_count: int, retained: list[tuple[int, int]]
    ) -> dict[str, object]:
        old_rows = list(self.rows)
        self.slot_count = slot_count
        self.rows = [None] * slot_count
        mapping = {}
        copied = 0
        for new_slot, (logical, old_slot) in enumerate(retained):
            self.rows[new_slot] = old_rows[old_slot]
            copied += len(self.rows[new_slot] or b"")
            mapping[logical] = new_slot
        return {"mapping": mapping, "bytes_copied": copied}


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


def test_fast_resize_copies_retained_rows_without_source_reads() -> None:
    class CountingSource(MemorySource):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        def get(self, layer: int, expert: int) -> ExpertPayload:
            self.reads += 1
            return super().get(layer, expert)

    source = CountingSource()
    sink = FastMemorySink(4)
    plane = ExpertSlotDataPlane(source)
    plane.register_layer(0, 8, sink, initial_experts=[0, 1, 2, 3])
    reads_before_resize = source.reads

    status = plane.resize(0, 2, retain=[2, 3])

    assert source.reads == reads_before_resize
    assert sink.rows == [bytes([2]) * 4, bytes([3]) * 4]
    assert status["counters"]["gpu_copy_bytes"] == 8
    assert status["counters"]["resize_time_ns"] > 0


def test_route_misses_are_loaded_as_one_source_batch() -> None:
    class BatchSource(MemorySource):
        def __init__(self) -> None:
            super().__init__()
            self.batches = []

        def get_many(self, requests: list[tuple[int, int]]):
            self.batches.append(list(requests))
            return [self.get(layer, expert) for layer, expert in requests]

    source = BatchSource()
    sink = MemorySink(4)
    plane = ExpertSlotDataPlane(source)
    plane.register_layer(0, 8, sink, initial_experts=[0, 1])

    mapping = plane.ensure(0, [0, 4, 5])

    assert set(mapping) == {0, 4, 5}
    assert source.batches[-1] == [(0, 4), (0, 5)]
    status = plane.layer_status(0)
    assert status["counters"]["batch_loads"] == 2
    assert status["counters"]["batch_objects"] == 4


def test_rdma_pool_stream_reuses_one_process_for_multiple_memory_gets(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "fake-pool"
    binary.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import struct
            import sys

            while True:
                raw = sys.stdin.buffer.read(4)
                if not raw:
                    break
                first = struct.unpack("!I", raw)[0]
                if first == 0:
                    break
                if first == 0xffffffff:
                    count = struct.unpack("!I", sys.stdin.buffer.read(4))[0]
                    keys = []
                    for _ in range(count):
                        size = struct.unpack("!I", sys.stdin.buffer.read(4))[0]
                        keys.append(sys.stdin.buffer.read(size))
                else:
                    keys = [sys.stdin.buffer.read(first)]
                for key in keys:
                    payload = b"warm:" + key
                    sys.stdout.buffer.write(struct.pack("!IQI", 0, len(payload), 0))
                    sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
            """
        ),
        encoding="utf-8",
    )
    binary.chmod(0o755)
    index = tmp_path / "profile.tsv"
    index.write_text(
        "0\tlayer-000/expert-0001.pllmex\t16\n"
        "1\tlayer-000/expert-0002.pllmex\t16\n",
        encoding="utf-8",
    )
    stream = RDMAPoolStream(
        "127.0.0.1",
        17902,
        binary,
        index,
        allocator="aligned",
    )

    first = stream.get("layer-000/expert-0001.pllmex")
    process = stream._process
    second = stream.get_many(
        ["layer-000/expert-0001.pllmex", "layer-000/expert-0002.pllmex"]
    )
    status = stream.status()
    stream.close()

    assert first == b"warm:layer-000/expert-0001.pllmex"
    assert second == [
        b"warm:layer-000/expert-0001.pllmex",
        b"warm:layer-000/expert-0002.pllmex",
    ]
    assert process is not None
    assert status["gets"] == 3
    assert status["persistent_qp"] is True
    assert status["local_disk_io"] is False


def test_rdma_pool_expert_store_decodes_without_installing_local_file(
    tmp_path: Path,
) -> None:
    local = SSDExpertStore(tmp_path / "cache", FINGERPRINT, required_format=FORMAT)
    payload = MemorySource().get(0, 3)
    encoded = ExpertPackageCodec.encode(payload)
    index = tmp_path / "profile.tsv"
    key = "layer-000/expert-0003.pllmex"
    index.write_text(f"3\t{key}\t{len(encoded)}\n", encoding="utf-8")

    class FakePool:
        index_file = index
        available = True

        def get(self, requested: str) -> bytes:
            assert requested == key
            return encoded

        def status(self):
            return {"persistent_qp": True, "local_disk_io": False}

    store = RDMAPoolExpertStore(FakePool(), local)

    restored = store.get(0, 3)

    assert restored == payload
    assert not local.path_for(0, 3).exists()
    assert store.status()["hot_path_sha256"] is False
