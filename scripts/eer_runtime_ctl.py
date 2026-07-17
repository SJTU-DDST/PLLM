from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pllm.vllm_eer_runtime import request_runtime


def main() -> None:
    runtime_dir = Path(os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    parser = argparse.ArgumentParser(description="Control the in-process vLLM EER data plane")
    parser.add_argument("command", choices=("status", "resize", "prefetch", "evict", "evict-all"))
    parser.add_argument("--socket", type=Path, default=runtime_dir / "pllm-eer.sock")
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--experts", default="")
    args = parser.parse_args()

    request: dict[str, object] = {"command": args.command.replace("-", "_")}
    if args.command == "resize":
        request.update({"slots_per_layer": args.slots, "quiesced": True})
    if args.command in {"prefetch", "evict"}:
        if args.layer is None:
            parser.error("--layer is required")
        request.update(
            {
                "layer": args.layer,
                "experts": [
                    int(item) for item in args.experts.split(",") if item.strip()
                ],
            }
        )
    if args.command == "evict-all":
        request["quiesced"] = True
    print(json.dumps(request_runtime(args.socket, request, timeout=600), indent=2))


if __name__ == "__main__":
    main()
