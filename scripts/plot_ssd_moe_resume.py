#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "naive_full_files": "Naive full\n20,480 files",
    "naive_full_pack": "Full\nsequential pack",
    "moe_hot_files": "MoE hot\nfiles",
    "moe_hot_pack": "MoE hot\nsequential pack",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SSD MoE resume benchmark")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    for name, result in payload["results"].items():
        rows.append(
            {
                "arm": name,
                "label": LABELS[name],
                "seconds": float(result["median_seconds"]),
                "gib": int(result["bytes"]) / 1024**3,
                "speedup_vs_files": float(result["speedup_vs_full_files"]),
                "speedup_vs_pack": float(result["speedup_vs_full_pack"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ssd_moe_resume.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    colors = ["#6b7280", "#4f7294", "#4b9277", "#25735e"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.7))
    x = list(range(len(rows)))
    axes[0].bar(x, [row["seconds"] for row in rows], color=colors)
    axes[0].set_ylabel("Cold restore seconds")
    axes[0].set_title("Local NVMe to memory")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [row["gib"] for row in rows], color=colors)
    axes[1].set_ylabel("GiB read")
    axes[1].set_title("Routed expert restore volume")
    axes[1].grid(axis="y", alpha=0.25)
    labels = [row["label"] for row in rows]
    for axis in axes:
        axis.set_xticks(x, labels)
    for index, row in enumerate(rows):
        axes[0].text(
            index,
            row["seconds"],
            f"{row['speedup_vs_files']:.2f}x",
            ha="center",
            va="bottom",
        )
        axes[1].text(
            index,
            row["gib"],
            f"{row['gib']:.1f}",
            ha="center",
            va="bottom",
        )
    figure.suptitle("SSD MoE selective resume (local NVMe)")
    figure.tight_layout()
    output = args.output_dir / "ssd_moe_resume.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
