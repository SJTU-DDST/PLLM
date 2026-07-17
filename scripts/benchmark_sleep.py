from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

try:
    import pynvml
except ImportError:
    pynvml = None


def gpu_memory_gb() -> float | None:
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
    except Exception:
        return None


def post(url: str, path: str, **kwargs) -> None:
    response = requests.post(f"{url}{path}", timeout=600, **kwargs)
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM sleep and wake")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mode", choices=("keep", "abort"), default="keep")
    parser.add_argument("--wake", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/sleep_bench.json"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    before = gpu_memory_gb()
    started = time.perf_counter()
    post(base, "/sleep", params={"level": args.level, "mode": args.mode})
    sleep_seconds = time.perf_counter() - started
    after = gpu_memory_gb()
    result = {
        "timestamp": time.time(),
        "level": args.level,
        "mode": args.mode,
        "sleep_seconds": sleep_seconds,
        "gpu_used_before_gb": before,
        "gpu_used_after_gb": after,
        "reclaimed_gb": before - after if before is not None and after is not None else None,
    }
    if args.wake:
        started = time.perf_counter()
        if args.level == 2:
            post(base, "/wake_up", params=[("tags", "weights")])
            post(base, "/collective_rpc", json={"method": "reload_weights"})
            post(base, "/wake_up", params=[("tags", "kv_cache")])
        else:
            post(base, "/wake_up")
        result["wake_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
