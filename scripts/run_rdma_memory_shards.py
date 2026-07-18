from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("RDMA pool client did not emit a JSON result")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sharded persistent-QP RDMA pool clients")
    parser.add_argument("--binary", type=Path, default=Path("rdma_bridge/build/pllm-rdma-pool"))
    parser.add_argument("--peer", required=True)
    parser.add_argument("--port", type=int, default=17902)
    parser.add_argument("--operation", choices=("put", "get"), required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allocator", choices=("aligned", "cuda-host"), default="cuda-host")
    parser.add_argument("--device", default="")
    parser.add_argument("--ib-port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=0)
    parser.add_argument("--queue-depth", type=int, default=32)
    parser.add_argument("--rd-atomic-depth", type=int, default=16)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    commands = []
    for index in args.index:
        command = [
            str(args.binary),
            "--client",
            args.peer,
            "--port",
            str(args.port),
            "--operation",
            args.operation,
            "--index",
            str(index),
            "--root",
            str(args.root),
            "--allocator",
            args.allocator,
            "--ib-port",
            str(args.ib_port),
            "--gid-index",
            str(args.gid_index),
            "--queue-depth",
            str(args.queue_depth),
            "--rd-atomic-depth",
            str(args.rd_atomic_depth),
            "--token-file",
            str(args.token_file),
        ]
        if args.device:
            command.extend(["--device", args.device])
        commands.append(command)

    started = time.perf_counter()
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for command in commands
    ]
    workers = []
    for process in processes:
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.strip() or stdout.strip())
        payload = parse_result(stdout)
        payload["progress"] = stderr.strip().replace("\r", " ").splitlines()[-1:]
        workers.append(payload)
    wall_seconds = time.perf_counter() - started
    total_bytes = sum(int(item["bytes"]) for item in workers)
    max_worker_seconds = max(float(item["seconds"]) for item in workers)
    max_rdma_seconds = max(float(item["rdma_seconds"]) for item in workers)
    result = {
        "schema_version": 1,
        "operation": args.operation,
        "workers": len(workers),
        "objects": sum(int(item["objects"]) for item in workers),
        "bytes": total_bytes,
        "wall_seconds": wall_seconds,
        "wall_effective_gbps": total_bytes * 8 / wall_seconds / 1e9,
        "max_worker_data_seconds": max_worker_seconds,
        "data_phase_effective_gbps": total_bytes
        * 8
        / max_worker_seconds
        / 1e9,
        "max_worker_rdma_seconds": max_rdma_seconds,
        "rdma_phase_effective_gbps": total_bytes
        * 8
        / max_rdma_seconds
        / 1e9,
        "allocator": args.allocator,
        "queue_depth": args.queue_depth,
        "rd_atomic_depth": args.rd_atomic_depth,
        "persistent_qp_per_worker": True,
        "one_sided": True,
        "remote_disk_io": False,
        "consistency_model": "phase_separated_epoch",
        "workers_detail": workers,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
