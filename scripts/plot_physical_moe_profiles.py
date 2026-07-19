#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_profile(specification: str) -> dict[str, Any]:
    slots_text, separator, path_text = specification.partition("=")
    if not separator:
        raise ValueError("profile must use SLOTS=PATH syntax")
    slots = int(slots_text)
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("results", [])
        if row.get("status") == "ok" and row.get("arm") == "resident_baseline"
    ]
    if not rows:
        raise ValueError(f"profile {path_text} has no successful resident baseline")
    return {
        "physical_slots_per_layer": slots,
        "rounds": len(rows),
        "mean_gpu_memory_mib": mean(
            float(row["memory_before_request_mib"])
            for row in rows
            if row.get("memory_before_request_mib") is not None
        ),
        "mean_tpot_ms": mean(float(row["all_tpot"]["mean_ms"]) for row in rows),
        "misses_per_text_event": sum(
            int(row["counter_delta"]["misses"]) for row in rows
        )
        / sum(int(row["text_events"]) for row in rows),
        "successful_output_rate": sum(bool(row.get("output")) for row in rows)
        / len(rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [str(row["physical_slots_per_layer"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    memory_bars = axes[0].bar(
        labels, [row["mean_gpu_memory_mib"] / 1024 for row in rows]
    )
    axes[0].bar_label(memory_bars, fmt="%.2f", padding=3)
    axes[0].set_ylabel("Measured GPU memory (GiB)")
    slowdown_bars = axes[1].bar(labels, [row["tpot_slowdown"] for row in rows])
    axes[1].bar_label(slowdown_bars, fmt="%.3fx", padding=3)
    axes[1].axhline(1.0, color="#667085", linestyle="--")
    axes[1].set_ylabel("Mean TPOT / largest profile TPOT")
    for axis in axes:
        axis.set_xlabel("Physical slots per MoE layer")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Static physical MoE residency: memory and decode cost")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare static physical MoE profiles from live benchmark outputs"
    )
    parser.add_argument(
        "--profile", action="append", required=True, help="SLOTS=benchmark.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = sorted(
        (load_profile(item) for item in args.profile),
        key=lambda row: int(row["physical_slots_per_layer"]),
        reverse=True,
    )
    baseline_tpot = rows[0]["mean_tpot_ms"]
    baseline_memory = rows[0]["mean_gpu_memory_mib"]
    for row in rows:
        row["tpot_slowdown"] = row["mean_tpot_ms"] / baseline_tpot
        row["measured_gpu_reclaim_mib"] = (
            baseline_memory - row["mean_gpu_memory_mib"]
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "physical_moe_profiles.csv", rows)
    plot(rows, args.output_dir / "physical_moe_profiles.png")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
