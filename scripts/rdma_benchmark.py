from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = ROOT / "rdma_bridge" / "build" / "pllm-rdma-stage"


def run_stage(binary: Path, size: int, allocator: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(binary), "--bytes", str(size), "--allocator", allocator],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    stream = result.stdout if result.returncode == 0 else result.stderr
    try:
        payload = json.loads(stream.strip().splitlines()[-1])
    except (ValueError, IndexError):
        payload = {"ready": False, "error": stream.strip() or "no benchmark output"}
    payload["returncode"] = result.returncode
    return payload


def run_network(peer: str, duration: int, device: str) -> dict[str, Any]:
    command = ["ib_write_bw", "-F", "--report_gbits", "-D", str(duration)]
    if device:
        command.extend(["-d", device])
    command.append(peer)
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=duration + 20, check=False
    )
    output = f"{result.stdout}\n{result.stderr}"
    rows = [line.strip() for line in output.splitlines() if re.match(r"^\d+\s+\d+", line.strip())]
    bandwidth = None
    if rows:
        fields = rows[-1].split()
        if len(fields) >= 4:
            try:
                bandwidth = float(fields[3])
            except ValueError:
                pass
    return {
        "attempted": True,
        "peer": peer,
        "device": device,
        "duration_seconds": duration,
        "bandwidth_gbps": bandwidth,
        "returncode": result.returncode,
        "tool": "ib_write_bw",
        "raw_tail": "\n".join(output.splitlines()[-12:]),
    }


def serve(duration: int, device: str) -> int:
    command = ["ib_write_bw", "-F", "--report_gbits", "-D", str(duration)]
    if device:
        command.extend(["-d", device])
    return subprocess.call(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="PLLM host-staged RDMA benchmark")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--allocator", choices=("aligned", "cuda-host"), default="aligned")
    parser.add_argument("--peer", default="")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rdma_bench.json")
    args = parser.parse_args()

    if args.server:
        raise SystemExit(serve(args.duration, args.device))
    if not args.binary.is_file():
        raise SystemExit(f"benchmark binary not found: {args.binary}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.time(),
        "host_staging": run_stage(args.binary, args.bytes, args.allocator),
        "network_rdma": {"attempted": False, "reason": "no peer configured"},
        "dgx_spark_theoretical_limit_gbps": 200,
        "claims": {"gpudirect_rdma": False, "metrics_are_separate": True},
    }
    if args.peer:
        payload["network_rdma"] = run_network(args.peer, args.duration, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
