from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm.expert_store import RDMABridgeTransport
from pllm.hiberstate import HiberStateStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or transfer committed PLLM live-state transactions"
    )
    parser.add_argument("command", choices=("status", "replicate", "fetch"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/ssd-storage/pllm-cache/live-state"),
    )
    parser.add_argument("--request-id", default="")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--model-fingerprint", default="")
    parser.add_argument("--peer", default="")
    parser.add_argument("--port", type=int, default=17901)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("rdma_bridge/build/pllm-rdma-store"),
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path.home() / ".config/pllm/rdma-token",
    )
    parser.add_argument(
        "--allocator", choices=("aligned", "cuda-host"), default="cuda-host"
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--ib-port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=0)
    args = parser.parse_args()

    transport = None
    if args.peer:
        transport = RDMABridgeTransport(
            args.peer,
            args.port,
            args.binary,
            token_file=args.token_file,
            timeout_seconds=120.0,
            allocator=args.allocator,
            device=args.device,
            ib_port=args.ib_port,
            gid_index=args.gid_index,
        )
    store = HiberStateStore(args.root, transport=transport)
    if args.command == "status":
        result = store.status()
    else:
        if not args.request_id or args.epoch is None:
            parser.error("replicate/fetch require --request-id and --epoch")
        if args.command == "replicate":
            store.replicate(args.request_id, args.epoch)
            result = {
                "replicated": True,
                "request_id": args.request_id,
                "epoch": args.epoch,
            }
        else:
            snapshot = store.fetch_remote(
                args.request_id,
                args.epoch,
                expected_model_fingerprint=args.model_fingerprint,
            )
            result = {
                "fetched": True,
                "request_id": snapshot.request_id,
                "epoch": snapshot.epoch,
                "committed_tokens": snapshot.committed_tokens,
                "components": [item.name for item in snapshot.components],
            }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
