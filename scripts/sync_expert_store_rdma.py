from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm.expert_store import (
    ExpertPackageCodec,
    RDMAExpertStore,
    SSDExpertStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate checksummed runtime experts to an RDMA warm source"
    )
    parser.add_argument("--root", type=Path, default=Path("/mnt/ssd-storage/pllm-experts"))
    parser.add_argument("--peer", required=True)
    parser.add_argument("--port", type=int, default=17900)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("rdma_bridge/build/pllm-rdma-store"),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".config/pllm/rdma-token")
    parser.add_argument("--allocator", choices=("aligned", "cuda-host"), default="aligned")
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((args.root / "runtime-manifest.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit("runtime expert export is incomplete")
    local = SSDExpertStore(
        args.root,
        model_fingerprint=str(manifest["model_fingerprint"]),
        required_format=str(manifest["format"]),
    )
    remote = RDMAExpertStore(
        args.peer,
        args.port,
        args.binary,
        local,
        token_file=args.token_file,
        allocator=args.allocator,
        device=args.device,
    )
    paths = sorted(args.root.glob("layer-*/expert-*.pllmex"))[args.start :]
    if args.limit > 0:
        paths = paths[: args.limit]
    for index, path in enumerate(paths, start=1):
        payload = ExpertPackageCodec.read(path)
        remote.put_path(path, payload)
        print(f"{index}/{len(paths)} {path.relative_to(args.root)}")
    if args.start == 0 and args.limit == 0:
        remote.transport.put(
            "runtime-manifest.json", args.root / "runtime-manifest.json"
        )
        print("committed remote runtime-manifest.json")


if __name__ == "__main__":
    main()
