#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import mmap
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pllm.config import DEFAULT_EXPERT_CACHE_DIR
from pllm.expert_store import MAGIC
from pllm.host_moe_resume import HostMoeResumePlan, plan_host_moe_resume
from pllm.ssd_resume_pack import ResumePackObject, build_resume_pack


def process_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key == "read_bytes":
            return int(value)
    return 0


def drop_file_cache(paths: list[Path]) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(
                descriptor, 0, 0, os.POSIX_FADV_DONTNEED
            )
        finally:
            os.close(descriptor)


def load_route_windows(
    route_dir: Path, history_steps: int, max_layers: int = 0
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    windows = []
    for path in sorted(route_dir.rglob("*.npz")):
        with np.load(path) as payload:
            decode = np.asarray(payload["decode"], dtype=np.int64)
        if max_layers > 0:
            decode = decode[:, :max_layers, :]
        for position in range(1, len(decode)):
            windows.append(
                (
                    str(path),
                    position,
                    decode[max(0, position - history_steps) : position],
                    decode[position],
                )
            )
    if not windows:
        raise ValueError("route directory contains no decode resume windows")
    return windows


def select_representative_plan(
    windows: list[tuple[str, int, np.ndarray, np.ndarray]],
    *,
    physical_slots: int,
    hot_slots: int,
    experts_per_layer: int,
) -> tuple[str, int, HostMoeResumePlan, dict[str, float]]:
    rows = []
    for path, position, history, next_routes in windows:
        plan = plan_host_moe_resume(
            history,
            next_routes,
            physical_slots=physical_slots,
            hot_slots=hot_slots,
            experts_per_layer=experts_per_layer,
        )
        rows.append((plan.exact_miss_objects, path, position, plan))
    rows.sort(key=lambda item: item[0])
    selected = rows[len(rows) // 2]
    misses = [row[0] for row in rows]
    p95_index = max(0, min(len(rows) - 1, int(len(rows) * 0.95)))
    return selected[1], selected[2], selected[3], {
        "windows": len(rows),
        "misses_mean": statistics.fmean(misses),
        "misses_p50": float(selected[0]),
        "misses_p95": float(misses[p95_index]),
    }


def all_expert_objects(
    cache_dir: Path, layer_ids: list[int], experts_per_layer: int
) -> list[ResumePackObject]:
    objects = []
    for layer in layer_ids:
        for expert in range(experts_per_layer):
            path = cache_dir / f"layer-{layer:03d}" / f"expert-{expert:04d}.pllmex"
            objects.append(ResumePackObject(layer, expert, path))
    return objects


def hot_expert_objects(
    cache_dir: Path, plan: HostMoeResumePlan, layer_ids: list[int]
) -> list[ResumePackObject]:
    objects = []
    for layer_index, (hot, misses) in enumerate(
        zip(plan.hot_experts_by_layer, plan.exact_misses_by_layer)
    ):
        layer = layer_ids[layer_index]
        selected = tuple(dict.fromkeys((*hot, *misses)))
        for expert in selected:
            path = cache_dir / f"layer-{layer:03d}" / f"expert-{expert:04d}.pllmex"
            objects.append(ResumePackObject(layer, expert, path))
    return objects


def discover_layer_ids(cache_dir: Path) -> list[int]:
    layer_ids = []
    for path in cache_dir.glob("layer-*"):
        if not path.is_dir():
            continue
        try:
            layer_ids.append(int(path.name.removeprefix("layer-")))
        except ValueError:
            continue
    if not layer_ids:
        raise FileNotFoundError(f"no expert layer directories found under {cache_dir}")
    return sorted(set(layer_ids))


def gpu_pressure(
    gib: float,
    seconds: float,
    result: dict[str, Any],
    ready: threading.Event,
) -> None:
    try:
        import torch

        total = int(gib * 1024**3)
        if total <= 0:
            ready.set()
            time.sleep(seconds)
            result.update({"bytes": 0, "touches": 0, "seconds": seconds})
            return
        allocation = torch.empty(total, dtype=torch.uint8, device="cuda")
        allocation[: min(total, 256 * 1024**2)].zero_()
        torch.cuda.synchronize()
        result["allocated_bytes"] = total
        ready.set()
        chunk = min(total, 256 * 1024**2)
        deadline = time.monotonic() + seconds
        touches = 0
        offset = 0
        while time.monotonic() < deadline:
            end = min(total, offset + chunk)
            allocation[offset:end].fill_(touches % 251)
            torch.cuda.synchronize()
            offset = 0 if end >= total else end
            touches += 1
        del allocation
        torch.cuda.empty_cache()
        result.update({"bytes": total, "touches": touches, "seconds": seconds})
    except Exception as exc:
        result["error"] = str(exc)
        ready.set()


def pack_objects(
    objects: list[ResumePackObject], path: Path
) -> tuple[dict[str, Any], list[int]]:
    started = time.perf_counter()
    read_before = process_read_bytes()
    manifest = build_resume_pack(objects, path)
    elapsed = time.perf_counter() - started
    read_bytes = process_read_bytes() - read_before
    offsets = [entry.offset for entry in manifest.entries]
    return {
        "objects": len(objects),
        "bytes": manifest.size_bytes,
        "seconds": elapsed,
        "source_read_bytes": read_bytes,
        "gib_per_second": manifest.size_bytes / 1024**3 / elapsed,
    }, offsets


def cold_restore(
    paths: list[Path],
    object_offsets: list[int],
    total_bytes: int,
    chunk_bytes: int,
) -> dict[str, Any]:
    drop_file_cache(paths)
    read_before = process_read_bytes()
    started = time.perf_counter()
    target = mmap.mmap(-1, total_bytes, access=mmap.ACCESS_WRITE)
    view = memoryview(target)
    cursor = 0
    try:
        for path in paths:
            with path.open("rb", buffering=0) as handle:
                remaining = path.stat().st_size
                while remaining:
                    size = min(chunk_bytes, remaining)
                    part = view[cursor : cursor + size]
                    read = handle.readinto(part)
                    part.release()
                    if read != size:
                        raise OSError(f"short restore read from {path}: {read}/{size}")
                    cursor += read
                    remaining -= read
        elapsed = time.perf_counter() - started
        if cursor != total_bytes:
            raise OSError(f"restore size mismatch: {cursor}/{total_bytes}")
        verified = sum(bytes(view[offset : offset + len(MAGIC)]) == MAGIC for offset in object_offsets)
        if verified != len(object_offsets):
            raise ValueError(f"expert package boundary check failed: {verified}/{len(object_offsets)}")
    finally:
        view.release()
        target.close()
    read_bytes = process_read_bytes() - read_before
    drop_file_cache(paths)
    gc.collect()
    return {
        "seconds": elapsed,
        "read_bytes": read_bytes,
        "verified_objects": verified,
        "gib_per_second": total_bytes / 1024**3 / elapsed,
    }


def file_layout(objects: list[ResumePackObject]) -> tuple[list[Path], list[int], int]:
    paths = []
    offsets = []
    cursor = 0
    for item in objects:
        paths.append(item.source)
        offsets.append(cursor)
        cursor += item.source.stat().st_size
    return paths, offsets, cursor


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SSD MoE selective resume")
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=Path("exp/moe_decode_locality_20260719/route_replay/routes"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.getenv("PLLM_EER_CACHE_DIR", str(DEFAULT_EXPERT_CACHE_DIR))
        ),
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        help="temporary pack directory; defaults to CACHE_DIR/.resume-packs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("exp/ssd_moe_resume"))
    parser.add_argument("--physical-slots", type=int, default=380)
    parser.add_argument("--hot-slots", type=int, default=256)
    parser.add_argument("--experts-per-layer", type=int, default=512)
    parser.add_argument("--history-steps", type=int, default=32)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="limit layers for a smoke test; zero uses every routed layer",
    )
    parser.add_argument("--pause-seconds", type=float, default=300.0)
    parser.add_argument("--pressure-gib", type=float, default=60.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--read-chunk-mib", type=int, default=64)
    parser.add_argument("--keep-packs", action="store_true")
    args = parser.parse_args()
    if (
        args.trials <= 0
        or args.pause_seconds < 0
        or args.read_chunk_mib <= 0
        or args.max_layers < 0
    ):
        parser.error("trials/chunk must be positive and pause cannot be negative")
    args.route_dir = args.route_dir.expanduser()
    args.cache_dir = args.cache_dir.expanduser()
    args.pack_dir = (
        args.pack_dir.expanduser()
        if args.pack_dir is not None
        else args.cache_dir / ".resume-packs"
    )
    args.output_dir = args.output_dir.expanduser()

    windows = load_route_windows(
        args.route_dir, args.history_steps, args.max_layers
    )
    route_path, route_position, plan, route_summary = select_representative_plan(
        windows,
        physical_slots=args.physical_slots,
        hot_slots=args.hot_slots,
        experts_per_layer=args.experts_per_layer,
    )
    layers = plan.layers
    layer_ids = discover_layer_ids(args.cache_dir)[:layers]
    if len(layer_ids) != layers:
        parser.error(
            f"route trace has {layers} layers but cache has {len(layer_ids)}"
        )
    full_objects = all_expert_objects(
        args.cache_dir, layer_ids, args.experts_per_layer
    )
    hot_objects = hot_expert_objects(args.cache_dir, plan, layer_ids)
    full_files, full_offsets, full_bytes = file_layout(full_objects)
    hot_files, hot_offsets, hot_bytes = file_layout(hot_objects)

    args.pack_dir.mkdir(parents=True, exist_ok=True)
    full_pack = args.pack_dir / "full-experts.pack"
    hot_pack = args.pack_dir / f"recent-{args.history_steps}-k{args.hot_slots}.pack"
    pressure_result: dict[str, Any] = {}
    pressure_ready = threading.Event()
    pressure_thread = threading.Thread(
        target=gpu_pressure,
        args=(
            args.pressure_gib,
            args.pause_seconds,
            pressure_result,
            pressure_ready,
        ),
        name="pllm-ssd-pause-pressure",
    )
    build_started = time.perf_counter()
    pressure_thread.start()
    try:
        if not pressure_ready.wait(timeout=60):
            raise TimeoutError("GPU pressure allocation did not become ready")
        if pressure_result.get("error"):
            raise RuntimeError(str(pressure_result["error"]))
        full_build, full_pack_offsets = pack_objects(full_objects, full_pack)
        hot_build, hot_pack_offsets = pack_objects(hot_objects, hot_pack)
        build_seconds = time.perf_counter() - build_started
        pressure_thread.join()
        if pressure_result.get("error"):
            raise RuntimeError(str(pressure_result["error"]))

        arms = {
            "naive_full_files": (full_files, full_offsets, full_bytes),
            "naive_full_pack": ([full_pack], full_pack_offsets, full_bytes),
            "moe_hot_files": (hot_files, hot_offsets, hot_bytes),
            "moe_hot_pack": ([hot_pack], hot_pack_offsets, hot_bytes),
        }
        measurements: dict[str, list[dict[str, Any]]] = {key: [] for key in arms}
        orders = [list(arms), list(reversed(arms))]
        chunk_bytes = args.read_chunk_mib * 1024**2
        for trial in range(args.trials):
            for name in orders[trial % len(orders)]:
                paths, offsets, size = arms[name]
                measurements[name].append(
                    cold_restore(paths, offsets, size, chunk_bytes)
                )

        results = {}
        for name, rows in measurements.items():
            seconds = [float(row["seconds"]) for row in rows]
            results[name] = {
                "bytes": arms[name][2],
                "objects": len(arms[name][1]),
                "trials": rows,
                "median_seconds": statistics.median(seconds),
                "median_gib_per_second": arms[name][2] / 1024**3 / statistics.median(seconds),
            }
        baseline = results["naive_full_files"]["median_seconds"]
        packed_baseline = results["naive_full_pack"]["median_seconds"]
        for row in results.values():
            row["speedup_vs_full_files"] = baseline / row["median_seconds"]
            row["speedup_vs_full_pack"] = packed_baseline / row["median_seconds"]

        payload = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "evidence": "local_nvme_to_anonymous_memory_exact_expert_packages",
            "configuration": {
                "route_dir": str(args.route_dir),
                "cache_dir": str(args.cache_dir),
                "pack_dir": str(args.pack_dir),
                "history_steps": args.history_steps,
                "physical_slots": args.physical_slots,
                "hot_slots": args.hot_slots,
                "experts_per_layer": args.experts_per_layer,
                "layers": layers,
                "layer_ids": layer_ids,
                "pause_seconds": args.pause_seconds,
                "pressure_gib": args.pressure_gib,
                "trials": args.trials,
                "read_chunk_mib": args.read_chunk_mib,
            },
            "selected_route": {
                "path": route_path,
                "position": route_position,
                **route_summary,
                "exact_route_covered": plan.exact_route_covered,
            },
            "pause": {
                "gpu_pressure": pressure_result,
                "pack_build_seconds": build_seconds,
                "pack_build_hidden_by_pause": build_seconds <= args.pause_seconds,
                "full_pack": full_build,
                "hot_pack": hot_build,
            },
            "results": results,
            "limitations": [
                "Runs on local NVMe and discrete RTX GPU; it is not a DGX Spark measurement.",
                "Restore target is anonymous system memory, matching UMA capacity traffic but not model execution.",
                "Measures routed expert packages only; dense/router/shared checkpoint bytes are excluded.",
                "Per-file cache is dropped with POSIX_FADV_DONTNEED without globally dropping system caches.",
            ],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "ssd_moe_resume.json"
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        if pressure_thread.is_alive():
            pressure_thread.join()
        if not args.keep_packs:
            for path in (
                full_pack,
                full_pack.with_suffix(full_pack.suffix + ".json"),
                hot_pack,
                hot_pack.with_suffix(hot_pack.suffix + ".json"),
            ):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
