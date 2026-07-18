from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
import torch


def action(api_base: str, name: str, level: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": name}
    if level is not None:
        payload["level"] = level
    response = requests.post(
        f"{api_base}/api/v1/actions", json=payload, timeout=900
    )
    response.raise_for_status()
    return response.json()


def try_allocation(gib: float) -> dict[str, Any]:
    requested_bytes = int(gib * 1024**3)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    allocation = None
    try:
        allocation = torch.empty(requested_bytes, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        return {
            "success": True,
            "requested_gib": gib,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
    except torch.OutOfMemoryError as exc:
        return {
            "success": False,
            "requested_gib": gib,
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(exc).splitlines()[0],
        }
    finally:
        del allocation
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure foreground CUDA allocation admission before and after PLLM"
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:17860")
    parser.add_argument("--allocation-gib", type=float, default=60.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/foreground_admission.json")
    )
    args = parser.parse_args()
    api_base = args.api_base.rstrip("/")

    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.time(),
        "evidence": "LIVE",
        "gpu": torch.cuda.get_device_name(0),
        "allocation_gib": args.allocation_gib,
        "resident": try_allocation(args.allocation_gib),
    }
    sleeping = False
    try:
        sleep_status = action(api_base, "hibernate", level=2)
        sleeping = True
        result["hibernate"] = {
            key: sleep_status.get(key)
            for key in (
                "state",
                "last_action_duration_ms",
                "reclaimed_gb",
                "sleep_level",
            )
        }
        result["released"] = try_allocation(args.allocation_gib)
    finally:
        if sleeping:
            wake_status = action(api_base, "wake")
            result["wake"] = {
                key: wake_status.get(key)
                for key in ("state", "last_action_duration_ms", "restore_source")
            }

    result["admission_restored"] = bool(
        not result["resident"]["success"] and result["released"]["success"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
