#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from pllm.config import DEFAULT_MODEL_PATH, PLLMConfig
from pllm.expert_catalog import ExpertCatalog
from pllm.expert_control import ExpertResidencyControlPlane


def synthetic_runtime(catalog: ExpertCatalog, profiles: list[int]) -> dict:
    metrics = {
        str(slots): {
            "per_layer": {
                str(layer): {
                    "byte_hit_rate_lower_bound": 0.999,
                    "mean_misses_per_token_upper_bound": (
                        0.001 if slots >= 500 else 0.01
                    ),
                    "p95_misses_per_token_upper_bound": 1.0,
                    "max_misses_per_token_upper_bound": 1.0,
                    "heldout_windows": 3,
                }
                for layer in catalog.moe_layers
            }
        }
        for slots in profiles
    }
    return {
        "route_trace": {
            "phase": "decode",
            "decode_observations": 40_960,
            "next_window": {
                "prediction_ready": True,
                "minimum_completed_windows": 4,
                "profiles": metrics,
            },
        },
        "decode_horizon": {"remaining_tokens": 1024, "decode_requests": 1},
        "data_plane": {
            "layers": [
                {"layer": layer, "slot_count": catalog.experts_per_layer}
                for layer in catalog.moe_layers
            ]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure synchronous Pareto solve and asynchronous dispatch cost"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/residency_planner_cpu.json")
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")

    catalog = ExpertCatalog.from_model(args.model_path)
    profiles = [256, 320, 384, 448, 480, 496, 504]
    runtime = synthetic_runtime(catalog, profiles)
    sync_ms = []
    peak_states = []
    for _ in range(args.iterations):
        control = ExpertResidencyControlPlane(
            PLLMConfig(
                model_path=str(args.model_path),
                decode_planner_async=False,
                expert_io_budget_gib_s=0.5,
            )
        )
        started = time.perf_counter()
        result = control.plan_decode_residency(runtime)
        sync_ms.append((time.perf_counter() - started) * 1000.0)
        peak_states.append(int(result.get("planner_peak_states", 0)))

    async_control = ExpertResidencyControlPlane(
        PLLMConfig(
            model_path=str(args.model_path),
            decode_planner_async=True,
            expert_io_budget_gib_s=0.5,
        )
    )
    started = time.perf_counter()
    pending = async_control.plan_decode_residency(runtime)
    dispatch_ms = (time.perf_counter() - started) * 1000.0
    while True:
        time.sleep(0.01)
        completed = async_control.plan_decode_residency(runtime)
        if not completed.get("planner_pending"):
            break
    async_solve_ms = (time.perf_counter() - started) * 1000.0

    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence": "cpu_synthetic_40_layer_planner_stress_not_route_quality",
        "model_path": str(args.model_path),
        "layers": len(catalog.moe_layers),
        "profiles": profiles,
        "iterations": args.iterations,
        "sync_wall_ms": sync_ms,
        "sync_p50_ms": statistics.median(sync_ms),
        "sync_max_ms": max(sync_ms),
        "peak_states": peak_states,
        "async_dispatch_ms": dispatch_ms,
        "async_pending_action": pending.get("action"),
        "async_solve_visible_ms": async_solve_ms,
        "completed_action": completed.get("action"),
        "gpu_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
