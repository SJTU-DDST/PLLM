#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from pllm.config import pllm_runtime_dir
from pllm.vllm_eer_runtime import request_runtime


def compact_route_mtp(status: dict[str, Any]) -> dict[str, Any]:
    route_mtp = status["route_mtp"]
    return {
        "backend": route_mtp["backend"],
        "mode": route_mtp["mode"],
        "observed_steps": route_mtp["observed_steps"],
        "predicted_steps": route_mtp["predicted_steps"],
        "mtp_signal_steps": route_mtp["mtp_signal_steps"],
        "mtp_signal_attached": route_mtp["mtp_signal_attached"],
        "last_required_uniform_rank": route_mtp["last_required_uniform_rank"],
        "coverage": route_mtp["coverage"],
        "queue_depth": route_mtp["queue_depth"],
        "enqueued_steps": route_mtp["enqueued_steps"],
        "dropped_steps": route_mtp["dropped_steps"],
        "critical_path": route_mtp["critical_path"],
        "gpu_probe": route_mtp["gpu_probe"],
        "eviction_enabled": route_mtp["eviction_enabled"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect paired RouteMTP and target routes from live vLLM"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="nvidia/nemotron-3-super")
    parser.add_argument("--socket", type=Path, default=pllm_runtime_dir() / "pllm-eer.sock")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default=(
            "Write a compact Python function that computes the longest increasing "
            "subsequence length and explain its complexity."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("max-tokens must be at least 2 for one-step pairing")

    request_runtime(
        args.socket,
        {"command": "phase", "phase": "decode", "reset_decode": True},
    )
    started = time.perf_counter()
    response = requests.post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        },
        timeout=900,
    )
    wall_seconds = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()

    status = None
    for _ in range(100):
        status = request_runtime(args.socket, {"command": "status"}, timeout=30)
        if int(status["route_mtp"]["queue_depth"]) == 0:
            break
        time.sleep(0.05)
    assert status is not None
    choice = payload["choices"][0]
    usage = payload.get("usage", {})
    result = {
        "schema_version": 1,
        "evidence": "live_vllm_paired_route_mtp_shadow",
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "wall_seconds": wall_seconds,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "finish_reason": choice.get("finish_reason"),
        "response_characters": len(choice.get("message", {}).get("content") or ""),
        "routed_experts_returned": bool(choice.get("routed_experts")),
        "runtime_slots_per_layer": status["slots_per_layer"],
        "runtime_minimum_slots_per_layer": status["minimum_slots_per_layer"],
        "runtime_maximum_slots_per_layer": status["maximum_slots_per_layer"],
        "route_mtp": compact_route_mtp(status),
        "notes": [
            "The target router remains authoritative and eviction is disabled.",
            "Target gates on MTP hidden are an uncalibrated initialization under test.",
            "The partial-resident profile may include blocking exact-route cache loads.",
        ],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["route_mtp"]["gpu_probe"]["paired_steps"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
