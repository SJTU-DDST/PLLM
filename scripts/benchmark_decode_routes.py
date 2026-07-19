#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from transformers import AutoTokenizer

from pllm.decode_residency import simulate_decode_cache
from pllm.expert_catalog import ExpertCatalog
from scripts.benchmark_longbench_qa import (
    DEFAULT_DATASETS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PATH,
    DatasetSpec,
    best_qa_f1,
    file_sha256,
    load_records,
    parse_dataset_spec,
    write_json,
)


DEFAULT_MISS_CURVE_MS = {
    1: 0.477,
    2: 0.784,
    4: 1.513,
    8: 2.863,
    16: 26.223,
    22: 42.710,
    32: 43.897,
}


def decode_routes(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded, validate=True)
    routes = np.load(io.BytesIO(raw), allow_pickle=False)
    if routes.ndim != 3:
        raise ValueError(f"unexpected routed-expert shape: {routes.shape}")
    return routes


def split_returned_routes(
    routes: np.ndarray,
    *,
    prompt_tokens: int,
    prompt_start: int,
    completion_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split vLLM 0.25.1's prompt chunk and later decode chunks."""
    prompt_rows = max(0, int(prompt_tokens) - int(prompt_start))
    decode_rows = max(0, int(completion_tokens) - 1)
    expected = prompt_rows + decode_rows
    if int(routes.shape[0]) != expected:
        raise ValueError(
            "routed-expert token layout mismatch: "
            f"received={routes.shape[0]}, expected={expected}, "
            f"prompt={prompt_tokens}, start={prompt_start}, "
            f"completion={completion_tokens}"
        )
    return routes[:prompt_rows], routes[prompt_rows:]


def batch_latency_ms(misses: float, curve: dict[int, float]) -> float:
    if misses <= 0:
        return 0.0
    points = sorted(curve.items())
    if misses <= points[0][0]:
        return points[0][1] * misses / points[0][0]
    for (left_n, left_ms), (right_n, right_ms) in zip(points, points[1:]):
        if misses <= right_n:
            ratio = (misses - left_n) / (right_n - left_n)
            return left_ms + ratio * (right_ms - left_ms)
    largest_n, largest_ms = points[-1]
    batches, remainder = divmod(misses, largest_n)
    return batches * largest_ms + (
        batch_latency_ms(remainder, curve) if remainder else 0.0
    )


def heldout_next_window_summary(
    route_arrays: list[np.ndarray],
    profiles: list[int],
    *,
    experts_per_layer: int,
    expert_bytes: int,
    window_tokens: int = 256,
    miss_curve_ms: dict[int, float] | None = None,
    baseline_tpot_ms: float = 100.0,
) -> dict[str, Any]:
    """Evaluate each past-window ranking only on the following route window."""
    arrays = [array for array in route_arrays if array.ndim == 3 and array.shape[0]]
    if not arrays:
        return {"available": False, "reason": "no decode routes"}
    request_windows = [
        [
            array[offset : offset + window_tokens]
            for offset in range(0, int(array.shape[0]) - window_tokens + 1, window_tokens)
        ]
        for array in arrays
    ]
    transitions = [
        (previous, following)
        for windows in request_windows
        for previous, following in zip(windows, windows[1:])
    ]
    decode_tokens = sum(int(array.shape[0]) for array in arrays)
    if not transitions:
        return {
            "available": False,
            "decode_tokens": decode_tokens,
            "reason": "no request contains two complete held-out windows",
        }
    curve = miss_curve_ms or DEFAULT_MISS_CURVE_MS
    layer_count = int(arrays[0].shape[1])
    results: dict[str, Any] = {}
    for slots in profiles:
        accesses = 0
        misses = 0
        token_misses: list[int] = []
        token_stall_ms: list[float] = []
        layer_misses: list[list[int]] = [[] for _ in range(layer_count)]
        for previous, following in transitions:
            hot_sets = []
            for layer in range(layer_count):
                counts = Counter(int(item) for item in previous[:, layer, :].reshape(-1))
                hot_sets.append(
                    set(
                        sorted(
                            range(experts_per_layer),
                            key=lambda expert: (-counts[expert], expert),
                        )[:slots]
                    )
                )
            for token in following:
                total = 0
                stall_ms = 0.0
                for layer, row in enumerate(token):
                    layer_total = sum(int(expert) not in hot_sets[layer] for expert in row)
                    layer_misses[layer].append(layer_total)
                    total += layer_total
                    stall_ms += batch_latency_ms(layer_total, curve)
                    accesses += len(row)
                misses += total
                token_misses.append(total)
                token_stall_ms.append(stall_ms)

        def p95(values: list[int]) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            return float(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)])

        def quantile(values: list[float], probability: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            return float(
                ordered[max(0, math.ceil(len(ordered) * probability) - 1)]
            )

        per_layer_p95 = [p95(values) for values in layer_misses]
        p95_latency = sum(batch_latency_ms(value, curve) for value in per_layer_p95)
        results[str(slots)] = {
            "slots_per_layer": slots,
            "heldout_transitions": len(transitions),
            "evaluated_tokens": len(token_misses),
            "expert_accesses": accesses,
            "blocking_misses": misses,
            "byte_hit_rate": (accesses - misses) / accesses if accesses else 0.0,
            "mean_misses_per_token": misses / max(1, len(token_misses)),
            "p95_misses_per_token": p95(token_misses),
            "per_layer_p95_misses": per_layer_p95,
            "miss_bytes": misses * expert_bytes,
            "p95_miss_latency_ms_per_token": p95_latency,
            "total_stall_ms_p50": quantile(token_stall_ms, 0.50),
            "total_stall_ms_p95": quantile(token_stall_ms, 0.95),
            "total_stall_ms_p99": quantile(token_stall_ms, 0.99),
            "miss_only_slowdown_ratio": (
                baseline_tpot_ms + quantile(token_stall_ms, 0.95)
            ) / baseline_tpot_ms,
            "resident_routed_bytes": int(
                round(
                    layer_count
                    * experts_per_layer
                    * expert_bytes
                    * slots
                    / experts_per_layer
                )
            ),
            "latency_estimator": (
                "quantile_of_per_token_sum_of_layer_batch_costs; network-source-only"
            ),
            "sum_of_layer_p95_ms_surrogate": p95_latency,
            "evidence": "past_window_rank_scored_only_on_following_window",
        }
    return {
        "available": True,
        "window_tokens": window_tokens,
        "complete_windows": sum(len(windows) for windows in request_windows),
        "requests_with_transitions": sum(
            len(windows) >= 2 for windows in request_windows
        ),
        "decode_tokens": decode_tokens,
        "profiles": results,
        "miss_latency_curve_ms": curve,
        "request_boundary_policy": "never_form_or_score_windows_across_requests",
    }


def heldout_layer_plan_summary(
    route_arrays: list[np.ndarray],
    slots_by_layer: dict[int, int],
    *,
    experts_per_layer: int,
    expert_bytes: int,
    window_tokens: int = 256,
    miss_curve_ms: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Replay one heterogeneous layer plan on request-local held-out tokens."""
    arrays = [array for array in route_arrays if array.ndim == 3 and array.shape[0]]
    if not arrays:
        return {"available": False, "reason": "no decode routes"}
    layer_count = int(arrays[0].shape[1])
    if set(slots_by_layer) != set(range(layer_count)):
        raise ValueError("slots_by_layer must define every routed layer index")
    if any(not 1 <= slots <= experts_per_layer for slots in slots_by_layer.values()):
        raise ValueError("layer plan contains an invalid slot count")
    transitions = []
    for array in arrays:
        windows = [
            array[offset : offset + window_tokens]
            for offset in range(
                0, int(array.shape[0]) - window_tokens + 1, window_tokens
            )
        ]
        transitions.extend(zip(windows, windows[1:]))
    if not transitions:
        return {
            "available": False,
            "reason": "no request contains two complete held-out windows",
        }

    curve = miss_curve_ms or DEFAULT_MISS_CURVE_MS
    accesses = 0
    misses = 0
    token_stall_ms: list[float] = []
    token_misses: list[int] = []
    for previous, following in transitions:
        hot_sets = []
        for layer in range(layer_count):
            counts = Counter(int(item) for item in previous[:, layer, :].reshape(-1))
            hot_sets.append(
                set(
                    sorted(
                        range(experts_per_layer),
                        key=lambda expert: (-counts[expert], expert),
                    )[: slots_by_layer[layer]]
                )
            )
        for token in following:
            token_total = 0
            stall = 0.0
            for layer, row in enumerate(token):
                layer_misses = sum(
                    int(expert) not in hot_sets[layer] for expert in row
                )
                token_total += layer_misses
                stall += batch_latency_ms(layer_misses, curve)
                accesses += len(row)
            misses += token_total
            token_misses.append(token_total)
            token_stall_ms.append(stall)

    def quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        return float(ordered[max(0, math.ceil(len(ordered) * probability) - 1)])

    tail = sorted(token_stall_ms)
    cvar_start = max(0, math.ceil(len(tail) * 0.95) - 1)
    return {
        "available": True,
        "slots_by_layer": {str(layer): slots for layer, slots in slots_by_layer.items()},
        "heldout_transitions": len(transitions),
        "evaluated_tokens": len(token_misses),
        "expert_accesses": accesses,
        "blocking_misses": misses,
        "object_hit_rate": (accesses - misses) / accesses if accesses else 0.0,
        "miss_bytes": misses * expert_bytes,
        "total_stall_ms_p50": quantile(token_stall_ms, 0.50),
        "total_stall_ms_p95": quantile(token_stall_ms, 0.95),
        "total_stall_ms_p99": quantile(token_stall_ms, 0.99),
        "total_stall_ms_cvar95": statistics.fmean(tail[cvar_start:]),
        "risk_estimator": "per_token_sum_of_layer_batch_source_costs",
        "evidence": "heterogeneous_plan_scored_on_request_local_next_windows",
    }


def run_sample(
    *,
    base_url: str,
    model: str,
    spec: DatasetSpec,
    sample_index: int,
    record: dict[str, Any],
    tokenizer: Any,
    catalog: ExpertCatalog,
    output_dir: Path,
    profiles: list[int],
    policies: list[str],
    prefix_tail_tokens: int,
    io_gib_s: float,
    baseline_tpot_ms: float,
    timeout: float,
) -> dict[str, Any]:
    local_prompt_tokens = len(
        tokenizer.encode(str(record["prompt"]), add_special_tokens=True)
    )
    prompt_start = max(0, local_prompt_tokens - prefix_tail_tokens)
    payload = {
        "model": model,
        "prompt": record["prompt"],
        "max_tokens": spec.max_tokens,
        "temperature": 0,
        "seed": 0,
        "vllm_xargs": {"routed_experts_prompt_start": prompt_start},
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/completions", json=payload, timeout=timeout
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    body = response.json()
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    encoded = choice.get("routed_experts")
    if not encoded:
        raise RuntimeError(
            "vLLM did not return routes; start with --enable-return-routed-experts "
            "and PLLM_DECODE_TRACE=1"
        )
    routes = decode_routes(str(encoded))
    moe_routes = routes[:, catalog.moe_layers, :].astype(np.uint16, copy=False)
    completion_tokens = int(usage.get("completion_tokens", 0))
    server_prompt_tokens = int(usage.get("prompt_tokens", local_prompt_tokens))
    prefill_tail, decode = split_returned_routes(
        moe_routes,
        prompt_tokens=server_prompt_tokens,
        prompt_start=prompt_start,
        completion_tokens=completion_tokens,
    )

    route_path = output_dir / "routes" / spec.name / f"{sample_index:04d}.npz"
    route_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        route_path,
        prefill_tail=prefill_tail,
        decode=decode,
        sample_id=str(record.get("_id", f"{spec.name}-{sample_index}")),
    )
    simulations = []
    for policy in policies:
        for slots in profiles:
            if decode.shape[0] == 0:
                continue
            result = simulate_decode_cache(
                prefill_tail,
                decode,
                slots,
                int(catalog.average_expert_bytes),
                policy=policy,
                experts_per_layer=catalog.experts_per_layer,
            ).to_dict()
            result["network_seconds_lower_bound"] = result["miss_bytes"] / (
                io_gib_s * 1024**3
            )
            misses_per_token = result["blocking_misses"] / max(
                1, result["decode_tokens"]
            )
            bandwidth_ms_per_token = (
                result["network_seconds_lower_bound"]
                / max(1, result["decode_tokens"])
                * 1000
            )
            average_layer_p95 = float(result["misses_per_token_p95"]) / max(
                1, decode.shape[1]
            )
            fixed_ms_per_token = decode.shape[1] * batch_latency_ms(
                average_layer_p95, DEFAULT_MISS_CURVE_MS
            )
            overhead_ms = max(bandwidth_ms_per_token, fixed_ms_per_token)
            result["misses_per_token"] = misses_per_token
            result["estimated_miss_ms_per_token"] = overhead_ms
            result["latency_estimator"] = (
                "average_layer_miss_count_on_cross_layer_rdma_p95_curve"
            )
            result["estimated_slowdown_ratio"] = (
                baseline_tpot_ms + overhead_ms
            ) / baseline_tpot_ms
            result.update(catalog.project_slots(slots))
            simulations.append(result)

    prediction = str(choice.get("text", "")).strip()
    references = [str(answer) for answer in record.get("answers", [])]
    return {
        "schema_version": 1,
        "dataset": spec.name,
        "sample_index": sample_index,
        "sample_id": record.get("_id", f"{spec.name}-{sample_index}"),
        "status": "ok",
        "prediction": prediction,
        "references": references,
        "f1": best_qa_f1(prediction, references),
        "latency_seconds": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "captured_prefill_tokens": int(prefill_tail.shape[0]),
        "captured_decode_tokens": int(decode.shape[0]),
        "captured_prompt_tail_tokens": int(prefill_tail.shape[0]),
        "server_prompt_tokens": server_prompt_tokens,
        "local_prompt_tokens": local_prompt_tokens,
        "routed_experts_prompt_start": prompt_start,
        "route_layout_evidence": (
            "vllm_0.25.1_scheduler_prompt_then_completion_minus_one"
        ),
        "route_shape": list(routes.shape),
        "route_file": str(route_path),
        "route_file_bytes": route_path.stat().st_size,
        "simulations": simulations,
    }


def summarize(rows: list[dict[str, Any]], profiles: list[int], policies: list[str]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    simulation_summary: dict[str, Any] = {}
    for policy in policies:
        for slots in profiles:
            selected = [
                simulation
                for row in successful
                for simulation in row.get("simulations", [])
                if simulation["policy"] == policy
                and int(simulation["slots_per_layer"]) == slots
            ]
            accesses = sum(int(item["expert_accesses"]) for item in selected)
            hits = sum(int(item["resident_hits"]) for item in selected)
            misses = sum(int(item["blocking_misses"]) for item in selected)
            miss_bytes = sum(int(item["miss_bytes"]) for item in selected)
            simulation_summary[f"{policy}:{slots}"] = {
                "policy": policy,
                "slots_per_layer": slots,
                "samples": len(selected),
                "expert_accesses": accesses,
                "resident_hits": hits,
                "blocking_misses": misses,
                "byte_hit_rate": hits / accesses if accesses else 0.0,
                "miss_gib": miss_bytes / 1024**3,
                "network_seconds_lower_bound": sum(
                    float(item["network_seconds_lower_bound"]) for item in selected
                ),
                "misses_per_token": misses
                / max(1, sum(int(item["decode_tokens"]) for item in selected)),
                "estimated_miss_ms_per_token": sum(
                    float(item["estimated_miss_ms_per_token"])
                    * int(item["decode_tokens"])
                    for item in selected
                )
                / max(1, sum(int(item["decode_tokens"]) for item in selected)),
                "estimated_slowdown_ratio": sum(
                    float(item["estimated_slowdown_ratio"])
                    * int(item["decode_tokens"])
                    for item in selected
                )
                / max(1, sum(int(item["decode_tokens"]) for item in selected)),
                "evidence": "offline_replay_of_live_full_resident_routes",
            }
    return {
        "completed_samples": len(successful),
        "errors": len(rows) - len(successful),
        "f1": sum(float(row["f1"]) for row in successful) / len(successful)
        if successful
        else 0.0,
        "wall_request_seconds": sum(
            float(row["latency_seconds"]) for row in successful
        ),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in successful),
        "completion_tokens": sum(
            int(row["completion_tokens"]) for row in successful
        ),
        "captured_decode_tokens": sum(
            int(row["captured_decode_tokens"]) for row in successful
        ),
        "route_file_bytes": sum(
            int(row["route_file_bytes"]) for row in successful
        ),
        "simulations": simulation_summary,
    }


def load_decode_route_arrays(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    arrays = []
    for row in rows:
        if row.get("status") != "ok" or not row.get("route_file"):
            continue
        with np.load(Path(row["route_file"]), allow_pickle=False) as payload:
            arrays.append(payload["decode"])
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture live full-resident routes and replay decode-only caches"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec)
    parser.add_argument("--profiles", default="256,320,384,448,480,496,504,512")
    parser.add_argument("--policies", default="lru,window_lfu")
    parser.add_argument("--prefix-tail-tokens", type=int, default=128)
    parser.add_argument("--io-gib-s", type=float, default=0.5)
    parser.add_argument("--baseline-tpot-ms", type=float, default=100.0)
    parser.add_argument("--heldout-window-tokens", type=int, default=256)
    parser.add_argument(
        "--slots-by-layer-json",
        type=Path,
        help="Optional heterogeneous planner output mapping model layer IDs to slots",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/decode_residency")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    profiles = sorted({int(item) for item in args.profiles.split(",")})
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    if (
        args.limit <= 0
        or args.prefix_tail_tokens <= 0
        or args.io_gib_s <= 0
        or args.baseline_tpot_ms <= 0
        or args.heldout_window_tokens <= 0
        or any(profile < 22 or profile > 512 for profile in profiles)
    ):
        parser.error("invalid limit, prefix tail, I/O bandwidth, or profile")

    catalog = ExpertCatalog.from_model(args.model_path)
    layer_plan: dict[int, int] | None = None
    if args.slots_by_layer_json is not None:
        raw_plan = json.loads(args.slots_by_layer_json.read_text(encoding="utf-8"))
        raw_plan = raw_plan.get("slots_by_layer", raw_plan)
        parsed = {int(layer): int(slots) for layer, slots in raw_plan.items()}
        if set(parsed) == set(catalog.moe_layers):
            layer_plan = {
                index: parsed[layer] for index, layer in enumerate(catalog.moe_layers)
            }
        elif set(parsed) == set(range(len(catalog.moe_layers))):
            layer_plan = parsed
        else:
            parser.error("slots-by-layer JSON must cover every routed model layer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    specs = args.dataset or [DatasetSpec(*dataset) for dataset in DEFAULT_DATASETS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "model_path": str(args.model_path),
        "profiles": profiles,
        "policies": policies,
        "prefix_tail_tokens": args.prefix_tail_tokens,
        "io_gib_s": args.io_gib_s,
        "baseline_tpot_ms": args.baseline_tpot_ms,
        "heldout_window_tokens": args.heldout_window_tokens,
        "miss_batch_p95_curve_ms": DEFAULT_MISS_CURVE_MS,
        "evidence": "live_full_resident_routes_with_offline_exact_cache_replay",
        "datasets": {},
    }

    selected_rows: list[dict[str, Any]] = []
    for spec in specs:
        records = load_records(spec.path, args.limit)
        output_path = args.output_dir / f"{spec.name}.jsonl"
        if args.overwrite:
            output_path.unlink(missing_ok=True)
        existing: dict[int, dict[str, Any]] = {}
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                existing[int(row["sample_index"])] = row
        with output_path.open("a", encoding="utf-8", buffering=1) as output:
            for index, record in enumerate(records):
                if existing.get(index, {}).get("status") == "ok":
                    continue
                try:
                    row = run_sample(
                        base_url=args.base_url,
                        model=args.model,
                        spec=spec,
                        sample_index=index,
                        record=record,
                        tokenizer=tokenizer,
                        catalog=catalog,
                        output_dir=args.output_dir,
                        profiles=profiles,
                        policies=policies,
                        prefix_tail_tokens=args.prefix_tail_tokens,
                        io_gib_s=args.io_gib_s,
                        baseline_tpot_ms=args.baseline_tpot_ms,
                        timeout=args.timeout,
                    )
                except Exception as exc:
                    row = {
                        "schema_version": 1,
                        "dataset": spec.name,
                        "sample_index": index,
                        "sample_id": record.get("_id", f"{spec.name}-{index}"),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[index] = row
                print(
                    f"[{spec.name}] {index + 1}/{len(records)} "
                    f"status={row['status']} f1={row.get('f1', 0):.4f}",
                    flush=True,
                )
        selected = [existing[index] for index in range(len(records))]
        selected_rows.extend(selected)
        dataset_summary = {
            "path": str(spec.path),
            "source_sha256": file_sha256(spec.path),
            **summarize(selected, profiles, policies),
        }
        dataset_summary["heldout_next_window"] = heldout_next_window_summary(
            load_decode_route_arrays(selected),
            profiles,
            experts_per_layer=catalog.experts_per_layer,
            expert_bytes=int(catalog.average_expert_bytes),
            window_tokens=args.heldout_window_tokens,
            baseline_tpot_ms=args.baseline_tpot_ms,
        )
        if layer_plan is not None:
            dataset_summary["heldout_layer_plan"] = heldout_layer_plan_summary(
                load_decode_route_arrays(selected),
                layer_plan,
                experts_per_layer=catalog.experts_per_layer,
                expert_bytes=int(catalog.average_expert_bytes),
                window_tokens=args.heldout_window_tokens,
            )
        summary["datasets"][spec.name] = dataset_summary
        write_json(args.output_dir / "summary.json", summary)

    summary["aggregate"] = summarize(selected_rows, profiles, policies)
    summary["aggregate"]["heldout_next_window"] = heldout_next_window_summary(
        load_decode_route_arrays(selected_rows),
        profiles,
        experts_per_layer=catalog.experts_per_layer,
        expert_bytes=int(catalog.average_expert_bytes),
        window_tokens=args.heldout_window_tokens,
        baseline_tpot_ms=args.baseline_tpot_ms,
    )
    if layer_plan is not None:
        summary["aggregate"]["heldout_layer_plan"] = heldout_layer_plan_summary(
            load_decode_route_arrays(selected_rows),
            layer_plan,
            experts_per_layer=catalog.experts_per_layer,
            expert_bytes=int(catalog.average_expert_bytes),
            window_tokens=args.heldout_window_tokens,
        )
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    return 0 if summary["aggregate"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
