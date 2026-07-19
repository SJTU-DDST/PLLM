#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("results", []) if row.get("status") == "ok"]
    if not rows:
        raise ValueError("live cycle output has no successful rows")
    return payload, rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("arm", f"pause_evict_{row['target_slots']}"))].append(row)
    baseline_rows = grouped.get("resident_baseline") or grouped.get("full_resident", [])
    if not baseline_rows:
        raise ValueError("live cycle output has no resident baseline")
    baseline_tpot = mean(
        float(row["all_tpot"]["mean_ms"])
        for row in baseline_rows
        if row.get("all_tpot", {}).get("mean_ms") is not None
    )
    result = []
    ordered_arms = sorted(
        grouped,
        key=lambda arm: (
            0 if arm in {"resident_baseline", "full_resident"} else 1 if arm == "pause_only" else 2,
            -int(grouped[arm][0]["target_slots"]),
        ),
    )
    for arm in ordered_arms:
        selected = grouped[arm]
        slots = int(selected[0]["target_slots"])
        post_values = [
            float(row["post_transition_tpot"]["mean_ms"])
            for row in selected
            if row.get("post_transition_tpot", {}).get("mean_ms") is not None
        ]
        if arm in {"resident_baseline", "full_resident"}:
            post_values = [
                float(row["all_tpot"]["mean_ms"])
                for row in selected
                if row.get("all_tpot", {}).get("mean_ms") is not None
            ]
        transitions = [
            float(row["transition"]["action_seconds"]) * 1000
            for row in selected
            if row.get("transition")
        ]
        memory_reclaim = [
            float(row["transition"]["memory_pre_resize_mib"])
            - float(row["transition"]["memory_post_resize_mib"])
            for row in selected
            if row.get("transition")
            and row["transition"].get("memory_pre_resize_mib") is not None
            and row["transition"].get("memory_post_resize_mib") is not None
        ]
        misses = [int(row["counter_delta"]["misses"]) for row in selected]
        text_events = [max(1, int(row["text_events"])) for row in selected]
        result.append(
            {
                "arm": arm,
                "slots_per_layer": slots,
                "rounds": len(selected),
                "mean_transition_ms": mean(transitions) if transitions else 0.0,
                "mean_transition_gap_ms": mean(
                    float(row["transition_gap_ms"])
                    for row in selected
                    if row.get("transition_gap_ms") is not None
                )
                if transitions
                else 0.0,
                "mean_memory_reclaim_mib": mean(memory_reclaim) if memory_reclaim else 0.0,
                "mean_tpot_ms": mean(post_values),
                "tpot_slowdown": mean(post_values) / baseline_tpot,
                "misses_per_text_event": sum(misses) / sum(text_events),
                "mean_bytes_loaded_mib": mean(
                    int(row["counter_delta"]["bytes_loaded"]) / 1024**2
                    for row in selected
                ),
                "successful_output_rate": sum(bool(row.get("output")) for row in selected)
                / len(selected),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], output: Path, residency_mode: str) -> None:
    labels = [
        "Baseline" if row["arm"] in {"resident_baseline", "full_resident"} else
        "Pause only" if row["arm"] == "pause_only" else
        str(row["slots_per_layer"])
        for row in rows
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes[0, 0].bar(labels, [row["mean_memory_reclaim_mib"] / 1024 for row in rows])
    axes[0, 0].set_ylabel("Measured GPU memory reclaim (GiB)")
    if residency_mode == "logical":
        axes[0, 0].text(
            0.5,
            0.5,
            "Logical eviction does not release weight tensors",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
    axes[0, 1].bar(labels, [row["mean_transition_gap_ms"] for row in rows])
    axes[0, 1].set_ylabel("Pause-to-next-token gap (ms)")
    axes[1, 0].plot(labels, [row["tpot_slowdown"] for row in rows], marker="o")
    axes[1, 0].axhline(1.0, color="#667085", linestyle="--")
    axes[1, 0].set_ylabel("Post-transition TPOT / resident baseline TPOT")
    axes[1, 1].plot(
        labels,
        [row["misses_per_text_event"] for row in rows],
        marker="o",
        color="#b42318",
    )
    axes[1, 1].set_ylabel("Whole-request expert misses per text event")
    for axis in axes.flat:
        axis.set_xlabel(
            "Active cache slots per MoE layer"
            if residency_mode == "logical"
            else "Physical slots per MoE layer"
        )
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Live token-boundary pause, logical MoE eviction, and wake"
        if residency_mode == "logical"
        else "Live token-boundary pause, physical MoE resize, and wake"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot live pause-MoE cycle performance")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, raw_rows = load_payload(args.input)
    rows = aggregate(raw_rows)
    write_csv(output_dir / "live_pause_moe_summary.csv", rows)
    plot(
        rows,
        output_dir / "live_pause_moe_performance.png",
        str(payload.get("residency_mode", "physical")),
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
