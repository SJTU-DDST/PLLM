from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def run_safe_benchmark(model_path: str, sample_mb: int = 64) -> dict[str, Any]:
    """Run bounded CPU and storage probes without creating a CUDA context."""
    sample_bytes = max(1, min(int(sample_mb), 256)) * 1024**2
    source = next(iter(sorted(Path(model_path).glob("*.safetensors"))), None)
    storage: dict[str, Any] = {"available": False}
    if source is not None:
        started = time.perf_counter()
        read_bytes = 0
        fd = os.open(source, os.O_RDONLY)
        try:
            offset = 0
            while read_bytes < sample_bytes:
                chunk = os.pread(fd, min(4 * 1024**2, sample_bytes - read_bytes), offset)
                if not chunk:
                    break
                read_bytes += len(chunk)
                offset += len(chunk)
        finally:
            os.close(fd)
        elapsed = max(time.perf_counter() - started, 1e-9)
        storage = {
            "available": True,
            "path": str(source),
            "bytes": read_bytes,
            "elapsed_ms": round(elapsed * 1000, 3),
            "bandwidth_gbps": round((read_bytes * 8) / elapsed / 1e9, 3),
            "cache_state": "uncontrolled",
        }

    payload = bytearray(min(sample_bytes, 64 * 1024**2))
    started = time.perf_counter()
    copied = bytes(payload)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "kind": "safe_cpu_storage",
        "gpu_allocated": False,
        "storage": storage,
        "host_copy": {
            "bytes": len(copied),
            "elapsed_ms": round(elapsed * 1000, 3),
            "bandwidth_gbps": round((len(copied) * 8) / elapsed / 1e9, 3),
        },
        "rdma_host_staging": _run_rdma_stage(),
    }


def _run_rdma_stage() -> dict[str, Any]:
    binary = Path(__file__).resolve().parent.parent / "rdma_bridge" / "build" / "pllm-rdma-stage"
    if not binary.is_file():
        return {"available": False, "reason": "rdma bridge is not built"}
    try:
        result = subprocess.run(
            [str(binary), "--allocator", "aligned", "--bytes", str(16 * 1024**2)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        stream = result.stdout if result.returncode == 0 else result.stderr
        payload = json.loads(stream.strip().splitlines()[-1])
        payload["available"] = result.returncode == 0
        return payload
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"available": False, "error": str(exc), "gpu_allocated": False}
