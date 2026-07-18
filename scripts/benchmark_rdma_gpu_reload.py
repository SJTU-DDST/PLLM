from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from pllm.expert_store import ExpertPackageCodec


def load_index(index: Path, root: Path) -> list[tuple[Path, int]]:
    rows = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        _slot, key, size = line.split("\t")
        rows.append((root / key, int(size)))
    if not rows:
        raise ValueError("profile index is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RDMA-reloaded expert packages and stage them into a GPU tensor"
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue-depth", type=int, default=16)
    parser.add_argument("--rdma-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queue_depth <= 0:
        raise ValueError("queue depth must be positive")

    rows = load_index(args.index, args.root)
    total_bytes = sum(size for _path, size in rows)
    max_bytes = max(size for _path, size in rows)
    device = torch.device("cuda:0")
    destination = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    staging = torch.empty(
        (args.queue_depth, max_bytes), dtype=torch.uint8, pin_memory=True
    )
    offsets = []
    cursor = 0
    for _path, size in rows:
        offsets.append(cursor)
        cursor += size

    validation_seconds = 0.0
    gpu_copy_seconds = 0.0
    sample_expected: dict[int, str] = {}
    started = time.perf_counter()
    for begin in range(0, len(rows), args.queue_depth):
        batch = rows[begin : begin + args.queue_depth]
        host_started = time.perf_counter()
        for item, (path, expected_size) in enumerate(batch):
            blob = path.read_bytes()
            if len(blob) != expected_size:
                raise ValueError(f"package size mismatch: {path}")
            ExpertPackageCodec.decode(blob)
            source = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
            staging[item, :expected_size].copy_(source)
            global_index = begin + item
            if global_index in {0, len(rows) - 1}:
                sample_expected[global_index] = hashlib.sha256(blob).hexdigest()
        validation_seconds += time.perf_counter() - host_started

        copy_started = time.perf_counter()
        for item, (_path, size) in enumerate(batch):
            global_index = begin + item
            offset = offsets[global_index]
            destination[offset : offset + size].copy_(
                staging[item, :size], non_blocking=True
            )
        torch.cuda.synchronize(device)
        gpu_copy_seconds += time.perf_counter() - copy_started

    stage_seconds = time.perf_counter() - started
    sample_results = []
    for index in (0, len(rows) - 1):
        path, size = rows[index]
        offset = offsets[index]
        content = destination[offset : offset + size].cpu().numpy().tobytes()
        digest = hashlib.sha256(content).hexdigest()
        sample_results.append(
            {
                "index": index,
                "key": path.relative_to(args.root).as_posix(),
                "sha256": digest,
                "matches": digest == sample_expected[index],
            }
        )
    if not all(item["matches"] for item in sample_results):
        raise RuntimeError("GPU sample verification failed")

    rdma_wall = 0.0
    if args.rdma_result:
        rdma_wall = float(
            json.loads(args.rdma_result.read_text(encoding="utf-8"))["wall_seconds"]
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "objects": len(rows),
        "bytes": total_bytes,
        "gpu": torch.cuda.get_device_name(device),
        "pinned_queue_depth": args.queue_depth,
        "host_read_and_sha_seconds": validation_seconds,
        "host_to_gpu_copy_seconds": gpu_copy_seconds,
        "host_to_gpu_effective_gbps": total_bytes * 8 / gpu_copy_seconds / 1e9,
        "gpu_stage_wall_seconds": stage_seconds,
        "rdma_reload_wall_seconds": rdma_wall,
        "rdma_to_gpu_wall_seconds": rdma_wall + stage_seconds,
        "rdma_to_gpu_wall_effective_gbps": (
            total_bytes * 8 / (rdma_wall + stage_seconds) / 1e9
            if rdma_wall
            else None
        ),
        "remote_disk_io": False,
        "gpudirect_claimed": False,
        "sample_verification": sample_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    del destination, staging
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
