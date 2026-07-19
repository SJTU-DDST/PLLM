#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pllm.config import DEFAULT_EXPERT_CACHE_DIR
from pllm.host_moe_resume import plan_host_moe_resume


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def parse_candidates(value: str) -> tuple[int, ...]:
    candidates = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not candidates or any(item <= 0 for item in candidates):
        raise argparse.ArgumentTypeError(
            "candidates must be positive comma-separated integers"
        )
    return candidates


def discover_expert_bytes(cache_dir: Path) -> int:
    sizes = [path.stat().st_size for path in cache_dir.rglob("*.pllmex")]
    if not sizes:
        raise FileNotFoundError(f"no runtime expert objects found under {cache_dir}")
    return int(statistics.median(sizes))


def load_windows(
    route_dir: Path, history_steps: int
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    windows = []
    for path in sorted(route_dir.rglob("*.npz")):
        with np.load(path) as payload:
            decode = np.asarray(payload["decode"], dtype=np.int64)
        if decode.ndim != 3:
            raise ValueError(f"decode route must be [tokens,layers,topk]: {path}")
        for position in range(1, len(decode)):
            history = decode[max(0, position - history_steps) : position]
            windows.append((str(path), position, history, decode[position]))
    if not windows:
        raise ValueError("route directory contains no resume windows")
    return windows


def scan_routes(
    windows: list[tuple[str, int, np.ndarray, np.ndarray]],
    candidates: tuple[int, ...],
    physical_slots: int,
    experts_per_layer: int,
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    misses_by_candidate: dict[int, list[int]] = {
        candidate: [] for candidate in candidates
    }
    rows = []
    for candidate in candidates:
        for _path, _position, history, next_routes in windows:
            plan = plan_host_moe_resume(
                history,
                next_routes,
                physical_slots=physical_slots,
                hot_slots=candidate,
                experts_per_layer=experts_per_layer,
            )
            if not plan.exact_route_covered:
                raise RuntimeError("host MoE plan did not cover an exact route")
            misses_by_candidate[candidate].append(plan.exact_miss_objects)
        misses = misses_by_candidate[candidate]
        naive_objects = physical_slots * len(next_routes)
        reductions = [
            1.0 - (candidate * len(next_routes) + miss) / naive_objects
            for miss in misses
        ]
        rows.append(
            {
                "hot_slots": candidate,
                "windows": len(misses),
                "exact_misses_mean": statistics.fmean(misses),
                "exact_misses_p95": percentile([float(item) for item in misses], 0.95),
                "expert_copy_reduction_mean": statistics.fmean(reductions),
                "expert_copy_reduction_p05": percentile(reductions, 0.05),
                "expert_copy_reduction_min": min(reductions),
            }
        )
    return rows, misses_by_candidate


def run_pressure(torch: Any, gib: float, seconds: float) -> dict[str, Any]:
    if gib <= 0:
        time.sleep(seconds)
        return {"bytes": 0, "touches": 0, "seconds": seconds}
    total = int(gib * 1024**3)
    allocation = torch.empty(total, dtype=torch.uint8, device="cuda")
    chunk = min(total, 256 * 1024**2)
    touches = 0
    deadline = time.monotonic() + seconds
    offset = 0
    while time.monotonic() < deadline:
        end = min(total, offset + chunk)
        allocation[offset:end].fill_(touches % 251)
        torch.cuda.synchronize()
        touches += 1
        offset = 0 if end >= total else end
    del allocation
    torch.cuda.empty_cache()
    return {"bytes": total, "touches": touches, "seconds": seconds}


def copy_timing(
    torch: Any,
    source: Any,
    destination: Any,
    *,
    dense_bytes: int,
    expert_bytes: int,
    hot_objects: int,
    miss_objects: int,
    naive: bool,
) -> float:
    if naive:
        sample_offsets = [dense_bytes, dense_bytes + (hot_objects - 1) * expert_bytes]
    else:
        copied_objects = hot_objects + miss_objects
        sample_offsets = [
            dense_bytes,
            dense_bytes + (hot_objects - 1) * expert_bytes,
            dense_bytes + (copied_objects - 1) * expert_bytes,
        ]
    sample_offsets = sorted(
        {offset for offset in sample_offsets if dense_bytes <= offset < len(source)}
    )
    expected = [int(source[offset].item()) for offset in sample_offsets]
    for offset in sample_offsets:
        destination[offset] = 0
    started = time.perf_counter()
    if naive:
        destination.copy_(source, non_blocking=True)
    else:
        critical = dense_bytes + hot_objects * expert_bytes
        destination[:critical].copy_(source[:critical], non_blocking=True)
        for index in range(miss_objects):
            begin = critical + index * expert_bytes
            end = begin + expert_bytes
            destination[begin:end].copy_(source[begin:end], non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    actual = [int(destination[offset].item()) for offset in sample_offsets]
    if actual != expected or any(value == 0 for value in expected):
        raise RuntimeError("host-to-GPU expert marker verification failed")
    return elapsed


def physical_benchmark(
    *,
    total_host_bytes: int,
    dense_bytes: int,
    expert_bytes: int,
    layers: int,
    physical_slots: int,
    selected_misses: dict[int, list[int]],
    trials: int,
    pause_seconds: float,
    pressure_gib: float,
) -> dict[str, Any]:
    import torch

    source = torch.empty(
        total_host_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
    )
    source.zero_()
    object_count = physical_slots * layers
    source[dense_bytes : dense_bytes + object_count * expert_bytes : expert_bytes] = (
        torch.arange(object_count, dtype=torch.int64) % 251 + 1
    ).to(torch.uint8)
    pause_pressure = run_pressure(torch, pressure_gib, pause_seconds)
    destination = torch.empty(total_host_bytes, dtype=torch.uint8, device="cuda")

    naive_times = []
    optimized: dict[str, Any] = {}
    for candidate, candidate_misses in selected_misses.items():
        misses = sorted(candidate_misses)
        scenarios = {
            "p50": misses[len(misses) // 2],
            "p95": int(percentile([float(item) for item in misses], 0.95)),
        }
        candidate_rows = {}
        for scenario, miss_objects in scenarios.items():
            times = []
            for trial in range(trials + 1):
                if trial % 2 == 0:
                    naive_elapsed = copy_timing(
                        torch,
                        source,
                        destination,
                        dense_bytes=dense_bytes,
                        expert_bytes=expert_bytes,
                        hot_objects=object_count,
                        miss_objects=0,
                        naive=True,
                    )
                    optimized_elapsed = copy_timing(
                        torch,
                        source,
                        destination,
                        dense_bytes=dense_bytes,
                        expert_bytes=expert_bytes,
                        hot_objects=candidate * layers,
                        miss_objects=miss_objects,
                        naive=False,
                    )
                else:
                    optimized_elapsed = copy_timing(
                        torch,
                        source,
                        destination,
                        dense_bytes=dense_bytes,
                        expert_bytes=expert_bytes,
                        hot_objects=candidate * layers,
                        miss_objects=miss_objects,
                        naive=False,
                    )
                    naive_elapsed = copy_timing(
                        torch,
                        source,
                        destination,
                        dense_bytes=dense_bytes,
                        expert_bytes=expert_bytes,
                        hot_objects=object_count,
                        miss_objects=0,
                        naive=True,
                    )
                if trial:
                    naive_times.append(naive_elapsed)
                    times.append(optimized_elapsed)
            critical_bytes = (
                dense_bytes + (candidate * layers + miss_objects) * expert_bytes
            )
            candidate_rows[scenario] = {
                "miss_objects": miss_objects,
                "critical_bytes": critical_bytes,
                "seconds": times,
                "median_seconds": statistics.median(times),
            }
        optimized[str(candidate)] = candidate_rows

    naive_median = statistics.median(naive_times)
    for candidate_rows in optimized.values():
        for row in candidate_rows.values():
            row["speedup_ratio"] = naive_median / row["median_seconds"]
            row["latency_reduction_ratio"] = 1.0 - row["median_seconds"] / naive_median

    del destination, source
    torch.cuda.empty_cache()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "host_pinned": True,
        "naive": {
            "bytes": total_host_bytes,
            "seconds": naive_times,
            "median_seconds": naive_median,
        },
        "optimized": optimized,
        "pause_pressure": pause_pressure,
        "verification": {
            "method": "nonzero per-object markers sampled after every transfer",
            "passed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark host-memory MoE priority resume"
    )
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=Path("exp/moe_decode_locality_20260719/route_replay/routes"),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EXPERT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("exp/host_moe_resume"))
    parser.add_argument("--physical-slots", type=int, default=380)
    parser.add_argument("--experts-per-layer", type=int, default=512)
    parser.add_argument("--history-steps", type=int, default=32)
    parser.add_argument(
        "--candidates",
        type=parse_candidates,
        default=parse_candidates("256,288,304,320"),
    )
    parser.add_argument("--total-host-gib", type=float, default=54.84)
    parser.add_argument("--pause-seconds", type=float, default=300.0)
    parser.add_argument("--pressure-gib", type=float, default=60.0)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials <= 0 or args.pause_seconds < 0 or args.total_host_gib <= 0:
        parser.error(
            "trials and total host size must be positive; pause cannot be negative"
        )
    if any(candidate > args.physical_slots for candidate in args.candidates):
        parser.error("candidate hot slots cannot exceed physical slots")

    windows = load_windows(args.route_dir, args.history_steps)
    route_rows, misses_by_candidate = scan_routes(
        windows,
        args.candidates,
        args.physical_slots,
        args.experts_per_layer,
    )
    layers = int(windows[0][3].shape[0])
    expert_bytes = discover_expert_bytes(args.cache_dir)
    total_host_bytes = int(args.total_host_gib * 1024**3)
    routed_bytes = args.physical_slots * layers * expert_bytes
    dense_bytes = total_host_bytes - routed_bytes
    if dense_bytes <= 0:
        parser.error("total host image is smaller than the routed expert pool")

    import torch

    physical = physical_benchmark(
        total_host_bytes=total_host_bytes,
        dense_bytes=dense_bytes,
        expert_bytes=expert_bytes,
        layers=layers,
        physical_slots=args.physical_slots,
        selected_misses=misses_by_candidate,
        trials=args.trials,
        pause_seconds=args.pause_seconds,
        pressure_gib=args.pressure_gib,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence": "local_pinned_host_memory_to_gpu_with_exact_route_fallback",
        "configuration": {
            "route_dir": str(args.route_dir),
            "cache_dir": str(args.cache_dir),
            "route_windows": len(windows),
            "history_steps": args.history_steps,
            "layers": layers,
            "experts_per_layer": args.experts_per_layer,
            "physical_slots": args.physical_slots,
            "candidate_hot_slots": list(args.candidates),
            "expert_bytes": expert_bytes,
            "routed_bytes": routed_bytes,
            "dense_bytes": dense_bytes,
            "total_host_bytes": total_host_bytes,
            "pause_seconds": args.pause_seconds,
            "pressure_gib": args.pressure_gib,
            "trials": args.trials,
        },
        "route_scan": route_rows,
        "pause_pressure": physical["pause_pressure"],
        "physical_h2d": physical,
        "limitations": [
            "Measures the selective Level-1 H2D critical path, not full model execution.",
            "The pinned byte carrier is size-matched to measured model weights; it does not execute those tensors.",
            "Host image is laid out dense-first and hot-expert-first during the pause window.",
            "Exact next-route misses are copied synchronously before the route can execute.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "host_moe_resume.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
