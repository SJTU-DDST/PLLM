#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import time
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


def decode_routes(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded, validate=True)
    routes = np.load(io.BytesIO(raw), allow_pickle=False)
    if routes.ndim != 3:
        raise ValueError(f"unexpected routed-expert shape: {routes.shape}")
    return routes


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
    miss_latency_p95_ms: float,
    baseline_tpot_ms: float,
    timeout: float,
) -> dict[str, Any]:
    local_prompt_tokens = len(
        tokenizer.encode(str(record["prompt"]), add_special_tokens=False)
    )
    prompt_start = max(0, local_prompt_tokens - prefix_tail_tokens - 8)
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
    decode_count = min(max(0, completion_tokens - 1), int(moe_routes.shape[0]))
    if decode_count:
        prefill_tail = moe_routes[:-decode_count]
        decode = moe_routes[-decode_count:]
    else:
        prefill_tail = moe_routes
        decode = moe_routes[:0]

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
            fixed_ms_per_token = misses_per_token * miss_latency_p95_ms
            overhead_ms = max(bandwidth_ms_per_token, fixed_ms_per_token)
            result["misses_per_token"] = misses_per_token
            result["estimated_miss_ms_per_token"] = overhead_ms
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture live full-resident routes and replay decode-only caches"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec)
    parser.add_argument("--profiles", default="384,448,480,496,504,512")
    parser.add_argument("--policies", default="lru,window_lfu")
    parser.add_argument("--prefix-tail-tokens", type=int, default=128)
    parser.add_argument("--io-gib-s", type=float, default=0.5)
    parser.add_argument("--miss-latency-p95-ms", type=float, default=7.5)
    parser.add_argument("--baseline-tpot-ms", type=float, default=100.0)
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
        or args.miss_latency_p95_ms <= 0
        or args.baseline_tpot_ms <= 0
        or any(profile < 22 or profile > 512 for profile in profiles)
    ):
        parser.error("invalid limit, prefix tail, I/O bandwidth, or profile")

    catalog = ExpertCatalog.from_model(args.model_path)
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
        "miss_latency_p95_ms": args.miss_latency_p95_ms,
        "baseline_tpot_ms": args.baseline_tpot_ms,
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
                        miss_latency_p95_ms=args.miss_latency_p95_ms,
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
        summary["datasets"][spec.name] = {
            "path": str(spec.path),
            "source_sha256": file_sha256(spec.path),
            **summarize(selected, profiles, policies),
        }
        write_json(args.output_dir / "summary.json", summary)

    summary["aggregate"] = summarize(selected_rows, profiles, policies)
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    return 0 if summary["aggregate"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
