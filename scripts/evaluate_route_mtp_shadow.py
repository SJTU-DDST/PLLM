#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pllm.config import DEFAULT_MODEL_PATH
from pllm.expert_catalog import ExpertCatalog
from pllm.route_forecast import RouteMTPResidencyPredictor


def evaluate_route_files(
    files: Iterable[Path],
    catalog: ExpertCatalog,
    candidates: Iterable[int],
    *,
    target_miss_rate: float = 0.05,
    confidence_delta: float = 0.01,
    minimum_samples: int = 128,
    include_layer_plans: bool = False,
) -> dict[str, Any]:
    selected = tuple(sorted(Path(path) for path in files))
    candidate_values = tuple(sorted({int(value) for value in candidates}))
    predictor = RouteMTPResidencyPredictor(
        catalog.moe_layers,
        catalog.experts_per_layer,
        catalog.active_experts_per_token,
        candidate_slots=candidate_values,
        target_miss_rate=target_miss_rate,
        confidence_delta=confidence_delta,
        minimum_calibration_samples=minimum_samples,
    )
    started = time.perf_counter()
    decoded_tokens = 0
    mtp_signal_rows = 0
    requests = 0
    skipped = []
    for path in selected:
        try:
            with np.load(path, allow_pickle=False) as payload:
                decode = payload["decode"]
                mtp_routes = payload["mtp_routes"] if "mtp_routes" in payload else None
        except (OSError, ValueError, KeyError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        if decode.ndim != 3 or decode.shape[1] != len(catalog.moe_layers):
            skipped.append(
                {
                    "path": str(path),
                    "reason": (
                        f"decode shape {list(decode.shape)} does not match "
                        f"{len(catalog.moe_layers)} MoE layers"
                    ),
                }
            )
            continue
        if mtp_routes is not None and (
            mtp_routes.ndim != 2 or mtp_routes.shape[0] != decode.shape[0]
        ):
            skipped.append(
                {
                    "path": str(path),
                    "reason": "mtp_routes must have shape [decode_tokens, top_k]",
                }
            )
            continue

        predictor.reset_request()
        requests += 1
        for token_index in range(decode.shape[0]):
            mtp_experts = (
                [int(value) for value in mtp_routes[token_index] if int(value) >= 0]
                if mtp_routes is not None
                else None
            )
            predictor.observe_step(
                {
                    layer: [
                        int(value)
                        for value in decode[token_index, layer_index]
                        if int(value) >= 0
                    ]
                    for layer_index, layer in enumerate(catalog.moe_layers)
                },
                mtp_experts=mtp_experts,
            )
            decoded_tokens += 1
            if mtp_experts:
                mtp_signal_rows += 1

    elapsed = time.perf_counter() - started
    status = predictor.status()
    plans = {}
    for slots in candidate_values:
        plan = predictor.residency_plan(slots)
        if not include_layer_plans:
            plan.pop("layers", None)
        plans[str(slots)] = plan
    return {
        "schema_version": 1,
        "evidence": (
            "cpu_route_replay_with_mtp_signal"
            if mtp_signal_rows
            else "cpu_route_replay_history_only_no_mtp_signal"
        ),
        "gpu_used": False,
        "files": len(selected),
        "requests": requests,
        "decode_tokens": decoded_tokens,
        "mtp_signal_rows": mtp_signal_rows,
        "wall_seconds": elapsed,
        "route_steps_per_second": decoded_tokens / elapsed if elapsed else 0.0,
        "skipped": skipped,
        "predictor": status,
        "plans": plans,
        "warning": (
            "This replay does not evaluate RouteMTP until traces contain mtp_routes."
            if not mtp_signal_rows
            else "MTP routes are observational signals; exact target Top-k remains authoritative."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only replay for the RouteMTP residency shadow planner"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--candidate-slots",
        default="256,320,384,448,480,496,504,512",
    )
    parser.add_argument("--target-miss-rate", type=float, default=0.05)
    parser.add_argument("--confidence-delta", type=float, default=0.01)
    parser.add_argument("--minimum-samples", type=int, default=128)
    parser.add_argument("--include-layer-plans", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidates = [
        int(value.strip())
        for value in args.candidate_slots.split(",")
        if value.strip()
    ]
    files: list[Path] = []
    for value in args.inputs:
        if value.is_dir():
            files.extend(sorted(value.rglob("*.npz")))
        else:
            files.append(value)
    result = evaluate_route_files(
        files,
        ExpertCatalog.from_model(args.model_path),
        candidates,
        target_miss_rate=args.target_miss_rate,
        confidence_delta=args.confidence_delta,
        minimum_samples=args.minimum_samples,
        include_layer_plans=args.include_layer_plans,
    )
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
