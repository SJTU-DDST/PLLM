#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm.config import DEFAULT_MODEL_PATH
from pllm.route_mtp import RouteMTPCheckpoint, inspect_vllm_route_mtp_support


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Nemotron MTP and build a no-allocation RouteMTP plan"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--include-tensors", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = RouteMTPCheckpoint.from_model(args.model_path)
    payload = checkpoint.summary()
    payload["router_probe"] = checkpoint.weight_plan().summary(
        include_tensors=args.include_tensors
    )
    payload["vllm"] = inspect_vllm_route_mtp_support().to_dict()
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
