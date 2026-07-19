from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any


def synchronize(torch: Any) -> None:
    torch.cuda.synchronize()


def drop_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def write_tensor(path: Path, tensor: Any) -> float:
    started = time.perf_counter()
    view = memoryview(tensor.numpy()).cast("B")
    offset = 0
    chunk_bytes = 64 * 1024**2
    with path.open("wb", buffering=0) as handle:
        while offset < len(view):
            written = handle.write(view[offset : offset + chunk_bytes])
            if written <= 0:
                raise OSError(f"short snapshot write at byte {offset}")
            offset += written
        os.fsync(handle.fileno())
    return time.perf_counter() - started


def read_tensor(path: Path, tensor: Any) -> float:
    started = time.perf_counter()
    view = memoryview(tensor.numpy()).cast("B")
    offset = 0
    chunk_bytes = 64 * 1024**2
    with path.open("rb", buffering=0) as handle:
        while offset < len(view):
            read = handle.readinto(view[offset : offset + chunk_bytes])
            if read <= 0:
                raise OSError(f"short snapshot read at byte {offset}")
            offset += read
    return time.perf_counter() - started


def release_gpu(torch: Any, *values: Any) -> None:
    for value in values:
        del value
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(torch)


def pressure_allocation(torch: Any, pressure_bytes: int) -> Any | None:
    if pressure_bytes <= 0:
        return None
    allocation = torch.empty(pressure_bytes, dtype=torch.uint8, device="cuda")
    allocation.zero_()
    synchronize(torch)
    return allocation


def allocate_kv(torch: Any, total_blocks: int, block_bytes: int, seed: int) -> Any:
    tensor = torch.zeros(
        (total_blocks, block_bytes), dtype=torch.uint8, device="cuda"
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return tensor, generator


def run_arm(
    torch: Any,
    arm: str,
    output_dir: Path,
    total_blocks: int,
    block_bytes: int,
    active_blocks: int,
    pause_seconds: float,
    pressure_bytes: int,
    seed: int,
) -> dict[str, Any]:
    synchronize(torch)
    torch.cuda.reset_peak_memory_stats()
    before_free, total_memory = torch.cuda.mem_get_info()
    gpu, generator = allocate_kv(torch, total_blocks, block_bytes, seed)
    active_ids = torch.arange(active_blocks, dtype=torch.long, device="cuda")
    gpu[:active_blocks].random_(0, 256, generator=generator)
    reference = gpu[:active_blocks].cpu()
    synchronize(torch)
    allocated_bytes = gpu.numel()
    result: dict[str, Any] = {
        "arm": arm,
        "pause_seconds": pause_seconds,
        "allocated_kv_bytes": allocated_bytes,
        "active_state_bytes": active_blocks * block_bytes,
        "active_blocks": active_blocks,
        "total_blocks": total_blocks,
        "block_bytes": block_bytes,
        "pressure_bytes": pressure_bytes,
        "gpu_total_bytes": total_memory,
        "gpu_free_before_bytes": before_free,
    }

    snapshot_started = time.perf_counter()
    snapshot_path = output_dir / f"{arm}.kv"
    host = None
    if arm == "keep_gpu":
        result["snapshot_bytes"] = 0
        result["gpu_reclaimed_bytes"] = 0
    elif arm == "full_ssd":
        host = gpu.cpu()
        synchronize(torch)
        result["device_to_host_seconds"] = time.perf_counter() - snapshot_started
        result["ssd_write_seconds"] = write_tensor(snapshot_path, host)
        result["snapshot_bytes"] = host.numel()
    elif arm in {"active_ssd", "active_cpu"}:
        host = gpu.index_select(0, active_ids).cpu()
        synchronize(torch)
        result["device_to_host_seconds"] = time.perf_counter() - snapshot_started
        result["snapshot_bytes"] = host.numel()
        if arm == "active_ssd":
            result["ssd_write_seconds"] = write_tensor(snapshot_path, host)
    else:
        raise ValueError(f"unknown arm: {arm}")
    result["snapshot_seconds"] = time.perf_counter() - snapshot_started

    pressure = None
    if arm != "keep_gpu":
        del gpu
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(torch)
        result["gpu_reclaimed_bytes"] = allocated_bytes
        if arm == "active_ssd":
            del host
            host = None
            gc.collect()
        pressure = pressure_allocation(torch, pressure_bytes)
        result["pressure_admitted"] = pressure is not None or pressure_bytes == 0
    else:
        free_with_kv, _ = torch.cuda.mem_get_info()
        result["pressure_admitted"] = pressure_bytes <= free_with_kv

    hold_started = time.perf_counter()
    time.sleep(pause_seconds)
    result["hold_seconds"] = time.perf_counter() - hold_started

    if pressure is not None:
        del pressure
        pressure = None
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(torch)

    restore_started = time.perf_counter()
    if arm == "keep_gpu":
        restored = gpu
    elif arm == "full_ssd":
        drop_file_cache(snapshot_path)
        restored_host = torch.empty((total_blocks, block_bytes), dtype=torch.uint8)
        result["ssd_read_seconds"] = read_tensor(snapshot_path, restored_host)
        restored = restored_host.to(device="cuda")
        synchronize(torch)
    elif arm == "active_ssd":
        drop_file_cache(snapshot_path)
        restored_host = torch.empty((active_blocks, block_bytes), dtype=torch.uint8)
        result["ssd_read_seconds"] = read_tensor(snapshot_path, restored_host)
        restored = torch.empty(
            (total_blocks, block_bytes), dtype=torch.uint8, device="cuda"
        )
        restored.index_copy_(0, active_ids, restored_host.to(device="cuda"))
        synchronize(torch)
    else:
        restored = torch.empty(
            (total_blocks, block_bytes), dtype=torch.uint8, device="cuda"
        )
        restored.index_copy_(0, active_ids, host.to(device="cuda"))
        synchronize(torch)
    result["restore_seconds"] = time.perf_counter() - restore_started
    result["exact_active_state"] = bool(torch.equal(restored[:active_blocks].cpu(), reference))
    result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    result["interruption_seconds"] = result["snapshot_seconds"] + result["restore_seconds"]
    result["snapshot_ratio"] = result["snapshot_bytes"] / allocated_bytes

    del restored, reference, active_ids
    if host is not None:
        del host
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(torch)
    if snapshot_path.exists():
        snapshot_path.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact KV pause strategies on the local GPU"
    )
    parser.add_argument("--allocated-gib", type=float, required=True)
    parser.add_argument("--capacity-tokens", type=int, required=True)
    parser.add_argument("--live-tokens", type=int, required=True)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--pause-seconds", type=float, default=300.0)
    parser.add_argument("--pressure-gib", type=float, default=0.0)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("keep_gpu", "full_ssd", "active_ssd", "active_cpu"),
        default=("keep_gpu", "full_ssd", "active_ssd", "active_cpu"),
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.live_tokens <= 0 or args.live_tokens > args.capacity_tokens:
        parser.error("live-tokens must be within (0, capacity-tokens]")
    if args.allocated_gib <= 0 or args.block_tokens <= 0:
        parser.error("allocated-gib and block-tokens must be positive")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_blocks = (args.capacity_tokens + args.block_tokens - 1) // args.block_tokens
    active_blocks = (args.live_tokens + args.block_tokens - 1) // args.block_tokens
    requested_bytes = int(args.allocated_gib * 1024**3)
    block_bytes = max(1, requested_bytes // total_blocks)
    pressure_bytes = int(args.pressure_gib * 1024**3)
    rows = []
    checkpoint = args.output_dir / "kv_pause_rows.jsonl"
    for arm in args.arms:
        row = run_arm(
            torch,
            arm,
            args.output_dir,
            total_blocks,
            block_bytes,
            active_blocks,
            args.pause_seconds,
            pressure_bytes,
            args.seed,
        )
        rows.append(row)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "created_at": time.time(),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "results": rows,
    }
    output = args.output_dir / "kv_pause_strategies.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)
    if not all(row["exact_active_state"] for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
