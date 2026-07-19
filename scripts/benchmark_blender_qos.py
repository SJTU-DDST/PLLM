#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PROFILES = {
    "quick": (1280, 720, 64),
    "demo": (1920, 1080, 256),
    "stress": (2560, 1440, 1024),
}


def api_status(base: str) -> dict[str, Any] | None:
    if not base:
        return None
    try:
        response = requests.get(f"{base.rstrip('/')}/api/v1/status", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one reproducible Blender Cycles QoS group"
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--blender", default="blender")
    parser.add_argument(
        "--project",
        type=Path,
        default=(
            Path(value)
            if (value := os.getenv("PLLM_BLENDER_PROJECT"))
            else None
        ),
    )
    parser.add_argument(
        "--profile-script",
        type=Path,
        default=(
            Path(value)
            if (value := os.getenv("PLLM_BLENDER_PROFILE_SCRIPT"))
            else None
        ),
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:17860")
    parser.add_argument("--output-dir", type=Path, default=Path("results/blender_qos"))
    args = parser.parse_args()
    if (
        args.project is None
        or args.profile_script is None
        or not args.project.is_file()
        or not args.profile_script.is_file()
    ):
        parser.error(
            "set --project/PLLM_BLENDER_PROJECT and "
            "--profile-script/PLLM_BLENDER_PROFILE_SCRIPT to readable files"
        )

    samples: list[dict[str, float]] = []
    stop = threading.Event()

    def monitor() -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop.wait(0.1):
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                samples.append(
                    {
                        "gpu_util": float(utilization.gpu),
                        "memory_mib": memory.used / 1024**2,
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                    }
                )
        except Exception:
            return

    before = api_status(args.api_base)
    command = [
        args.blender,
        "-b",
        str(args.project),
        "--python",
        str(args.profile_script),
        "-f",
        "1",
        "--",
        args.profile,
    ]
    thread = threading.Thread(target=monitor, name="pllm-blender-nvml", daemon=True)
    thread.start()
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    stop.set()
    thread.join(timeout=2)
    after = api_status(args.api_base)
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-4000:] or process.stdout[-4000:])

    width, height, render_samples = PROFILES[args.profile]
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence": "live_blender_cycles_render",
        "label": args.label,
        "profile": args.profile,
        "resolution": [width, height],
        "render_samples": render_samples,
        "wall_seconds": wall_seconds,
        "sample_pixels_per_second": width * height * render_samples / wall_seconds,
        "gpu_samples": len(samples),
        "gpu_util_mean": statistics.fmean(item["gpu_util"] for item in samples)
        if samples
        else None,
        "gpu_util_peak": max((item["gpu_util"] for item in samples), default=None),
        "gpu_memory_peak_mib": max(
            (item["memory_mib"] for item in samples), default=None
        ),
        "power_mean_w": statistics.fmean(item["power_w"] for item in samples)
        if samples
        else None,
        "power_peak_w": max((item["power_w"] for item in samples), default=None),
        "pllm_before": before,
        "pllm_after": after,
        "command": command,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.label}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
