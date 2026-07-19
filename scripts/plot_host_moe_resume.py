#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot host-memory MoE resume results")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    naive = payload["physical_h2d"]["naive"]
    rows = [
        {
            "label": "Naive full host",
            "hot_slots": int(payload["configuration"]["physical_slots"]),
            "scenario": "full",
            "seconds": float(naive["median_seconds"]),
            "speedup": 1.0,
            "bytes": int(naive["bytes"]),
        }
    ]
    for slots, scenarios in payload["physical_h2d"]["optimized"].items():
        for scenario, result in scenarios.items():
            rows.append(
                {
                    "label": f"Recent-32 K={slots} ({scenario})",
                    "hot_slots": int(slots),
                    "scenario": scenario,
                    "seconds": float(result["median_seconds"]),
                    "speedup": float(result["speedup_ratio"]),
                    "bytes": int(result["critical_bytes"]),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "host_moe_resume.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    selected = [rows[0]] + [row for row in rows[1:] if row["scenario"] == "p50"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    colors = ["#6b7280"] + ["#2f7d68"] * (len(selected) - 1)
    axes[0].bar(
        range(len(selected)), [row["seconds"] for row in selected], color=colors
    )
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Host memory to executable route")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        range(len(selected)),
        [row["bytes"] / 1024**3 for row in selected],
        color=colors,
    )
    axes[1].set_ylabel("GiB copied before first route")
    axes[1].set_title("Critical H2D volume")
    axes[1].grid(axis="y", alpha=0.25)
    labels = [row["label"].replace("Recent-32 ", "") for row in selected]
    for axis in axes:
        axis.set_xticks(range(len(selected)), labels, rotation=18, ha="right")
    for index, row in enumerate(selected):
        axes[0].text(
            index,
            row["seconds"],
            f"{row['speedup']:.2f}x",
            ha="center",
            va="bottom",
        )
        axes[1].text(
            index,
            row["bytes"] / 1024**3,
            f"{row['bytes'] / 1024**3:.1f}",
            ha="center",
            va="bottom",
        )
    figure.suptitle("Pinned host-memory MoE priority resume (local GPU)")
    figure.tight_layout()
    output = args.output_dir / "host_moe_resume.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
