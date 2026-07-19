#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pllm.decode_residency import simulate_decode_cache
from pllm.expert_catalog import ExpertCatalog


def replay_profile(arguments: tuple[list[str], int, int, int, int]) -> dict[str, Any]:
    paths, slots, pin_steps, expert_bytes, experts_per_layer = arguments
    hits = misses = accesses = tokens = 0
    for path_text in paths:
        with np.load(path_text) as payload:
            result = simulate_decode_cache(
                payload["prefill_tail"],
                payload["decode"],
                slots,
                expert_bytes,
                policy="window_lfu",
                history_window=64,
                experts_per_layer=experts_per_layer,
                protect_recent_tokens=pin_steps,
            )
        hits += result.resident_hits
        misses += result.blocking_misses
        accesses += result.expert_accesses
        tokens += result.decode_tokens
    return {
        "slots_per_layer": slots,
        "pin_recent_steps": pin_steps,
        "decode_tokens": tokens,
        "expert_accesses": accesses,
        "byte_hit_rate": hits / accesses if accesses else 0.0,
        "blocking_misses": misses,
        "misses_per_token": misses / tokens if tokens else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay captured MoE routes across recent-pin strategies"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--profiles", default="380,382,383,384")
    parser.add_argument("--pin-steps", default="0,8,32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary_path = args.input_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_path = args.model_path or Path(summary["model_path"])
    catalog = ExpertCatalog.from_model(model_path)
    profiles = sorted({int(item) for item in args.profiles.split(",")})
    pin_steps = sorted({int(item) for item in args.pin_steps.split(",")})
    paths = sorted(str(path) for path in (args.input_dir / "routes").glob("*/*.npz"))
    if (
        not paths
        or args.workers <= 0
        or any(profile < catalog.active_experts_per_token for profile in profiles)
        or any(profile > catalog.experts_per_layer for profile in profiles)
        or any(steps < 0 for steps in pin_steps)
    ):
        parser.error("invalid routes, workers, profiles, or pin steps")
    work = [
        (
            paths,
            profile,
            steps,
            int(catalog.average_expert_bytes),
            catalog.experts_per_layer,
        )
        for profile in profiles
        for steps in pin_steps
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(replay_profile, work))
    for row in rows:
        row.update(catalog.project_slots(int(row["slots_per_layer"])))
    rows.sort(key=lambda row: (int(row["pin_recent_steps"]), int(row["slots_per_layer"])))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence": "offline_replay_of_captured_exact_routes",
        "model_path": str(model_path),
        "route_files": len(paths),
        "rows": rows,
    }
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recent_pin_route_scan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "recent_pin_route_scan.csv", rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
