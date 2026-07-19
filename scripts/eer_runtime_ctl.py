from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm.config import pllm_runtime_dir
from pllm.vllm_eer_runtime import request_runtime


def main() -> None:
    runtime_dir = pllm_runtime_dir()
    parser = argparse.ArgumentParser(description="Control the in-process vLLM EER data plane")
    parser.add_argument(
        "command",
        choices=(
            "status",
            "resize",
            "set-capacity",
            "set-pin-recent-steps",
            "prefetch",
            "evict",
            "evict-all",
            "suspend",
            "resume",
        ),
    )
    parser.add_argument("--socket", type=Path, default=runtime_dir / "pllm-eer.sock")
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--pin-recent-steps", type=int, default=32)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--experts", default="")
    args = parser.parse_args()

    request: dict[str, object] = {"command": args.command.replace("-", "_")}
    if args.command in {"resize", "set-capacity"}:
        request.update({"slots_per_layer": args.slots, "quiesced": True})
    if args.command == "set-pin-recent-steps":
        request["steps"] = args.pin_recent_steps
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
    if args.command in {"evict-all", "suspend"}:
        request["quiesced"] = True
    print(json.dumps(request_runtime(args.socket, request, timeout=600), indent=2))


if __name__ == "__main__":
    main()
