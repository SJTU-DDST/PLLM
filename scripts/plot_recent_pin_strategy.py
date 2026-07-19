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


def load_run(specification: str) -> list[dict[str, Any]]:
    label, separator, path_text = specification.partition("=")
    if not separator:
        raise ValueError("run must use LABEL=PATH syntax")
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    successful = [
        row for row in payload.get("results", []) if row.get("status") == "ok"
    ]
    baseline = [row for row in successful if row.get("arm") == "resident_baseline"]
    if not baseline:
        raise ValueError(f"run {path_text} has no resident baseline")
    baseline_tpot = mean(float(row["all_tpot"]["mean_ms"]) for row in baseline)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in successful:
        arm = str(row.get("arm", ""))
        if arm == "pause_only":
            continue
        grouped.setdefault(arm, []).append(row)
    rows = []
    for arm, selected in grouped.items():
        slots = int(selected[0]["target_slots"])
        if arm == "resident_baseline":
            tpot_values = [float(row["all_tpot"]["mean_ms"]) for row in selected]
        else:
            tpot_values = [
                float(row["post_transition_tpot"]["mean_ms"])
                for row in selected
            ]
        rows.append(
            {
                "strategy": label,
                "slots_per_layer": slots,
                "rounds": len(selected),
                "mean_tpot_ms": mean(tpot_values),
                "tpot_slowdown": mean(tpot_values) / baseline_tpot,
                "misses_per_text_event": sum(
                    int(row["counter_delta"]["misses"]) for row in selected
                )
                / sum(int(row["text_events"]) for row in selected),
            }
        )
    return sorted(rows, key=lambda row: int(row["slots_per_layer"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    strategies = list(dict.fromkeys(str(row["strategy"]) for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for strategy in strategies:
        selected = [row for row in rows if row["strategy"] == strategy]
        slots = [int(row["slots_per_layer"]) for row in selected]
        axes[0].plot(
            slots,
            [float(row["tpot_slowdown"]) for row in selected],
            marker="o",
            label=strategy,
        )
        axes[1].plot(
            slots,
            [float(row["misses_per_text_event"]) for row in selected],
            marker="o",
            label=strategy,
        )
    axes[0].axhline(1.0, color="#667085", linestyle="--")
    axes[0].set_ylabel("Post-transition TPOT / same-run baseline")
    axes[1].set_ylabel("Whole-request expert misses per text event")
    for axis in axes:
        axis.set_xlabel("Active cache slots per MoE layer")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Recent-route pinning improves pause-MoE residency")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare normalized live results for recent-route pin strategies"
    )
    parser.add_argument("--run", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for item in args.run for row in load_run(item)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "recent_pin_strategy.csv", rows)
    plot(rows, args.output_dir / "recent_pin_strategy.png")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
