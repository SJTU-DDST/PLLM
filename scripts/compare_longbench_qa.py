#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def compare_dataset(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_f1": baseline["f1"],
        "candidate_f1": candidate["f1"],
        "f1_delta": candidate["f1"] - baseline["f1"],
        "baseline_total_tokens_per_second": baseline["total_tokens_per_second"],
        "candidate_total_tokens_per_second": candidate["total_tokens_per_second"],
        "throughput_ratio": ratio(
            candidate["total_tokens_per_second"], baseline["total_tokens_per_second"]
        ),
        "baseline_completion_tokens_per_second": baseline[
            "completion_tokens_per_second"
        ],
        "candidate_completion_tokens_per_second": candidate[
            "completion_tokens_per_second"
        ],
        "completion_throughput_ratio": ratio(
            candidate["completion_tokens_per_second"],
            baseline["completion_tokens_per_second"],
        ),
        "baseline_latency_p50": baseline["latency_seconds_p50"],
        "candidate_latency_p50": candidate["latency_seconds_p50"],
        "latency_p50_ratio": ratio(
            candidate["latency_seconds_p50"], baseline["latency_seconds_p50"]
        ),
        "baseline_gpu_memory_peak_mib": baseline["gpu"].get("gpu_memory_mib_peak"),
        "candidate_gpu_memory_peak_mib": candidate["gpu"].get("gpu_memory_mib_peak"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    names = sorted(set(baseline["datasets"]) & set(candidate["datasets"]))
    comparison = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline_mode": baseline["mode"],
        "candidate_mode": candidate["mode"],
        "datasets": {
            name: compare_dataset(baseline["datasets"][name], candidate["datasets"][name])
            for name in names
        },
        "aggregate": {
            "baseline_weighted_f1": baseline["aggregate"]["sample_weighted_f1"],
            "candidate_weighted_f1": candidate["aggregate"]["sample_weighted_f1"],
            "f1_delta": candidate["aggregate"]["sample_weighted_f1"]
            - baseline["aggregate"]["sample_weighted_f1"],
            "baseline_total_tokens_per_second": baseline["aggregate"][
                "total_tokens_per_second"
            ],
            "candidate_total_tokens_per_second": candidate["aggregate"][
                "total_tokens_per_second"
            ],
            "throughput_ratio": ratio(
                candidate["aggregate"]["total_tokens_per_second"],
                baseline["aggregate"]["total_tokens_per_second"],
            ),
            "baseline_completion_tokens_per_second": baseline["aggregate"].get(
                "completion_tokens_per_second"
            ),
            "candidate_completion_tokens_per_second": candidate["aggregate"].get(
                "completion_tokens_per_second"
            ),
            "completion_throughput_ratio": ratio(
                candidate["aggregate"].get("completion_tokens_per_second", 0.0),
                baseline["aggregate"].get("completion_tokens_per_second", 0.0),
            ),
        },
        "storage": {
            "baseline_before": baseline["storage_before"],
            "baseline_after": baseline["storage_after"],
            "candidate_before": candidate["storage_before"],
            "candidate_after": candidate["storage_after"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
