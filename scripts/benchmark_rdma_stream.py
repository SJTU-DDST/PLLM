#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from pllm.expert_store import ExpertPackageCodec, RDMAPoolStream


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def read_index(index_path: Path) -> list[tuple[str, int]]:
    objects = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        _slot, key, expected_size = line.split("\t")
        objects.append((key, int(expected_size)))
    if not objects:
        raise ValueError("profile is empty")
    return objects


def read_profile(
    rows: list[tuple[str, int]], root: Path, selected_keys: set[str]
) -> dict[str, bytes]:
    objects = {}
    for key, expected_size in rows:
        if key not in selected_keys:
            continue
        content = (root / key).read_bytes()
        if len(content) != expected_size:
            raise ValueError(f"profile size mismatch: {key}")
        ExpertPackageCodec.decode(content)
        objects[key] = content
    if len(objects) != len(selected_keys):
        raise ValueError("one or more selected keys are absent from the profile")
    return objects


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark persistent-QP RDMA memory-pool stream GET"
    )
    parser.add_argument("--peer", required=True)
    parser.add_argument("--port", type=int, default=17902)
    parser.add_argument(
        "--binary", type=Path, default=Path("rdma_bridge/build/pllm-rdma-pool")
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--gid-index", type=int, default=0)
    parser.add_argument("--allocator", choices=("aligned", "cuda-host"), default="aligned")
    parser.add_argument("--shared-staging", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=22)
    parser.add_argument(
        "--selection", choices=("strided", "sequential"), default="strided"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/rdma_stream_get_probe.json")
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0 or args.batch_size > 32:
        parser.error("iterations must be positive and batch size must be within [1, 32]")

    rows = read_index(args.index)
    keys = [key for key, _size in rows]
    if args.selection == "strided":
        # Both strides are co-prime to the 20,480-object production image, so
        # adjacent batch entries span layers and successive batches rotate the
        # complete image deterministically.
        requested_batches = [
            [
                keys[(iteration * 997 + offset * 929) % len(keys)]
                for offset in range(args.batch_size)
            ]
            for iteration in range(args.iterations)
        ]
    else:
        requested_batches = [
            [
                keys[(iteration * args.batch_size + offset) % len(keys)]
                for offset in range(args.batch_size)
            ]
            for iteration in range(args.iterations)
        ]
    expected = read_profile(
        rows,
        args.root,
        {key for batch in requested_batches for key in batch},
    )
    stream = RDMAPoolStream(
        args.peer,
        args.port,
        args.binary,
        args.index,
        token_file=args.token_file,
        allocator=args.allocator,
        device=args.device,
        gid_index=args.gid_index,
        timeout_seconds=60,
        batch_size=args.batch_size,
        shared_staging=args.shared_staging,
    )
    latencies_ms: list[float] = []
    batch_bytes: list[int] = []
    transferred = 0
    started = time.perf_counter()
    try:
        for requested in requested_batches:
            get_started = time.perf_counter()
            actual = stream.get_many(requested)
            latencies_ms.append((time.perf_counter() - get_started) * 1000)
            current_bytes = 0
            for key, content in zip(requested, actual):
                if content != expected[key]:
                    raise ValueError(
                        f"remote bytes differ from the source profile: {key}"
                    )
                transferred += len(content)
                current_bytes += len(content)
            batch_bytes.append(current_bytes)
        status = stream.status()
    finally:
        stream.close()
    elapsed = time.perf_counter() - started
    steady = latencies_ms[1:] or latencies_ms
    steady_bytes = sum(batch_bytes[1:] or batch_bytes)
    steady_seconds = sum(steady) / 1000.0
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence": "live_rocev2_remote_volatile_memory_pool",
        "peer": args.peer,
        "port": args.port,
        "profile_objects": len(rows),
        "validated_source_objects": len(expected),
        "batches": args.iterations,
        "batch_size": args.batch_size,
        "selection": args.selection,
        "shared_staging": args.shared_staging,
        "objects": args.iterations * args.batch_size,
        "bytes": transferred,
        "wall_seconds": elapsed,
        "wall_effective_gbps": transferred * 8 / elapsed / 1e9,
        "steady_effective_gbps": (
            steady_bytes * 8 / steady_seconds / 1e9 if steady_seconds else 0.0
        ),
        "first_get_ms": latencies_ms[0],
        "steady_mean_ms": statistics.fmean(steady),
        "steady_p50_ms": percentile(steady, 0.50),
        "steady_p95_ms": percentile(steady, 0.95),
        "steady_p99_ms": percentile(steady, 0.99),
        "byte_exact": True,
        "source_packages_sha256_validated_before_timing": True,
        "remote_disk_io": False,
        "local_destination_disk_io": False,
        "transport": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
