#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = (
    "naive_full_files",
    "naive_full_pack",
    "moe_hot_files",
    "moe_hot_pack",
)
LABELS = {
    "naive_full_files": "Full set / files",
    "naive_full_pack": "Full set / pack",
    "moe_hot_files": "MoE hot set / files",
    "moe_hot_pack": "MoE hot set / pack",
}
GROUPS = (
    ("Full expert set\n(20,480 objects)", ORDER[:2]),
    ("MoE hot set\n(10,421 objects)", ORDER[2:]),
)
SERIES = (
    ("Individual files", "#597DA3", ""),
    ("Sequential pack", "#D07A52", "////"),
)


def configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SSD MoE resume benchmark")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload["results"]
    rows = []
    for name in ORDER:
        result = results[name]
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

    configure_paper_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    group_centers = [0.0, 1.0]
    width = 0.34
    offsets = (-width / 2, width / 2)

    for series_index, (series_label, color, hatch) in enumerate(SERIES):
        names = [group[1][series_index] for group in GROUPS]
        positions = [center + offsets[series_index] for center in group_centers]
        medians = [float(results[name]["median_seconds"]) for name in names]
        trials = [
            [float(row["seconds"]) for row in results[name]["trials"]]
            for name in names
        ]
        lower = [median - min(values) for median, values in zip(medians, trials)]
        upper = [max(values) - median for median, values in zip(medians, trials)]
        latency_bars = axes[0].bar(
            positions,
            medians,
            width=width * 0.9,
            label=series_label,
            color=color,
            edgecolor="#252525",
            linewidth=0.65,
            hatch=hatch,
            yerr=[lower, upper],
            error_kw={"ecolor": "#252525", "elinewidth": 0.7, "capsize": 2},
            zorder=3,
        )
        volumes = [int(results[name]["bytes"]) / 1024**3 for name in names]
        volume_bars = axes[1].bar(
            positions,
            volumes,
            width=width * 0.9,
            color=color,
            edgecolor="#252525",
            linewidth=0.65,
            hatch=hatch,
            zorder=3,
        )
        for bar, name, median in zip(latency_bars, names, medians):
            speedup = float(results[name]["speedup_vs_full_files"])
            axes[0].annotate(
                f"{median:.1f}\n{speedup:.2f}x",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                linespacing=0.9,
            )
        for bar, volume in zip(volume_bars, volumes):
            axes[1].annotate(
                f"{volume:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    group_labels = [group[0] for group in GROUPS]
    for axis in axes:
        axis.set_xticks(group_centers, group_labels)
        axis.grid(axis="y", color="#B8B8B8", linestyle="--", linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.1)
    axes[0].set_title("(a) Cold-resume latency")
    axes[0].set_ylabel("Resume latency (s)")
    axes[0].set_ylim(0, 69)
    axes[0].legend(loc="upper right", ncols=1)
    axes[0].text(
        0.02,
        0.97,
        "Median of 3 trials; min-max error bars",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#404040",
    )
    axes[1].set_title("(b) SSD read volume")
    axes[1].set_ylabel("Data read (GiB)")
    axes[1].set_ylim(0, 69)

    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.91, wspace=0.28)
    png_output = args.output_dir / "ssd_moe_resume.png"
    pdf_output = args.output_dir / "ssd_moe_resume.pdf"
    figure.savefig(png_output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(pdf_output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(png_output)
    print(pdf_output)


if __name__ == "__main__":
    main()
