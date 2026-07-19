#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from pllm.expert_store import ExpertPackageCodec, RDMAPoolStream


def load_index(path: Path) -> list[tuple[str, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        _slot, key, size = line.split("\t")
        rows.append((key, int(size)))
    if not rows:
        raise ValueError("profile index is empty")
    return rows


def strided_sample(rows: list[tuple[str, int]], count: int) -> list[tuple[str, int]]:
    count = min(count, len(rows))
    stride = 997
    while math.gcd(stride, len(rows)) != 1:
        stride += 2
    return [rows[(index * stride) % len(rows)] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure direct shared-MR package parse and host-to-GPU staging"
    )
    parser.add_argument("--peer", default=os.getenv("PLLM_EER_RDMA_PEER"))
    parser.add_argument("--port", type=int, default=17902)
    parser.add_argument(
        "--binary", type=Path, default=Path("rdma_bridge/build/pllm-rdma-pool")
    )
    parser.add_argument(
        "--index", type=Path, default=Path("results/eer-memory-profile-full.tsv")
    )
    parser.add_argument(
        "--token-file", type=Path, default=Path("~/.config/pllm/rdma-token")
    )
    parser.add_argument("--device", default="mlx5_0")
    parser.add_argument("--gid-index", type=int, default=5)
    parser.add_argument("--objects", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=Path("results/rdma_to_gpu_pipeline.json")
    )
    args = parser.parse_args()
    if not args.peer:
        parser.error("set --peer or PLLM_EER_RDMA_PEER")
    if args.objects <= 0 or not 1 <= args.batch_size <= 32:
        parser.error("objects must be positive and batch size must be within [1, 32]")

    rows = strided_sample(load_index(args.index), args.objects)
    total_bytes = sum(size for _key, size in rows)
    offsets = []
    cursor = 0
    for _key, size in rows:
        offsets.append(cursor)
        cursor += size

    transport = RDMAPoolStream(
        peer=args.peer,
        port=args.port,
        binary=args.binary,
        index_file=args.index,
        token_file=args.token_file.expanduser(),
        device=args.device,
        gid_index=args.gid_index,
        batch_size=args.batch_size,
        shared_staging=True,
        cuda_register_staging=True,
    )
    destination = torch.empty(total_bytes, dtype=torch.uint8, device="cuda:0")
    # QP/MR setup is intentionally outside the steady pipeline measurement.
    transport.get(rows[0][0])

    rdma_seconds = 0.0
    parse_seconds = 0.0
    h2d_seconds = 0.0
    copied = 0
    expected_samples: dict[int, str] = {}
    iterator = iter(transport.iter_many([key for key, _size in rows]))
    started = time.perf_counter()
    while copied < len(rows):
        phase_started = time.perf_counter()
        blobs = next(iterator)
        rdma_seconds += time.perf_counter() - phase_started

        parse_started = time.perf_counter()
        payloads = [ExpertPackageCodec.decode(blob, verify=False) for blob in blobs]
        parse_seconds += time.perf_counter() - parse_started

        copy_started = time.perf_counter()
        for offset, (blob, payload) in enumerate(zip(blobs, payloads)):
            index = copied + offset
            expected_size = rows[index][1]
            if len(blob) != expected_size:
                raise ValueError(f"package size mismatch at object {index}")
            if (payload.layer, payload.expert) != (
                int(rows[index][0].split("/")[0].split("-")[1]),
                int(rows[index][0].split("/")[1].split("-")[1].split(".")[0]),
            ):
                raise ValueError(f"package identity mismatch at object {index}")
            source = torch.frombuffer(blob, dtype=torch.uint8)
            destination[offsets[index] : offsets[index] + expected_size].copy_(
                source, non_blocking=True
            )
            if index in {0, len(rows) - 1}:
                expected_samples[index] = hashlib.sha256(blob).hexdigest()
        torch.cuda.synchronize()
        h2d_seconds += time.perf_counter() - copy_started
        copied += len(blobs)
    iterator.close()
    wall_seconds = time.perf_counter() - started

    verification = []
    for index in (0, len(rows) - 1):
        size = rows[index][1]
        content = destination[offsets[index] : offsets[index] + size].cpu().numpy()
        digest = hashlib.sha256(content.tobytes()).hexdigest()
        verification.append(
            {
                "index": index,
                "key": rows[index][0],
                "matches": digest == expected_samples[index],
            }
        )
    if not all(item["matches"] for item in verification):
        raise RuntimeError("GPU sample verification failed")

    status = transport.status()
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence": "live_remote_shared_host_mr_to_flat_gpu_byte_carrier",
        "objects": len(rows),
        "bytes": total_bytes,
        "batch_size": args.batch_size,
        "selection": "strided_from_full_index",
        "gpu": torch.cuda.get_device_name(0),
        "rdma_get_seconds": rdma_seconds,
        "package_parse_seconds": parse_seconds,
        "host_to_gpu_seconds": h2d_seconds,
        "pipeline_wall_seconds": wall_seconds,
        "pipeline_effective_gbps": total_bytes * 8 / wall_seconds / 1e9,
        "host_to_gpu_effective_gbps": total_bytes * 8 / h2d_seconds / 1e9,
        "transport": status,
        "sample_verification": verification,
        "remote_disk_io": False,
        "local_disk_io": False,
        "gpudirect_claimed": False,
        "limitations": (
            "flat package-byte destination; excludes Marlin tensor placement, "
            "mapping publication, and kernel rebuild"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    transport.close()
    del destination
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
