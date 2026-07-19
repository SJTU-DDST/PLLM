from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "keep_gpu": "Keep KV on GPU",
    "full_ssd": "Full KV via SSD",
    "active_ssd": "Live blocks via SSD",
    "active_cpu": "Live blocks in CPU",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local KV pause benchmark")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    rows = [row for payload in payloads for row in payload["results"]]
    order = {name: index for index, name in enumerate(LABELS)}
    rows.sort(key=lambda row: order[row["arm"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = args.output_dir / "kv_pause_strategies_combined.json"
    combined_path.write_text(
        json.dumps(
            {
                "device": payloads[-1].get("device", ""),
                "source_files": [str(path) for path in args.inputs],
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = args.output_dir / "kv_pause_strategies.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = [LABELS[row["arm"]] for row in rows]
    entries = [row["snapshot_seconds"] for row in rows]
    wakes = [row["restore_seconds"] for row in rows]
    sizes = [row["snapshot_bytes"] / 1024**3 for row in rows]
    x = list(range(len(rows)))

    figure, (latency_axis, size_axis) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    latency_axis.bar(x, entries, label="Pause entry", color="#5b8ff9")
    latency_axis.bar(x, wakes, bottom=entries, label="Wake restore", color="#61d9a3")
    latency_axis.set_ylabel("Seconds (5-minute hold excluded)")
    latency_axis.set_xticks(x, labels, rotation=18, ha="right")
    latency_axis.legend(frameon=False)
    latency_axis.grid(axis="y", alpha=0.25)

    size_axis.bar(x, sizes, color="#d97757")
    size_axis.set_ylabel("KV snapshot GiB")
    size_axis.set_xticks(x, labels, rotation=18, ha="right")
    size_axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(sizes):
        size_axis.text(index, value, f"{value:.2f}", ha="center", va="bottom")

    figure.suptitle("Exact KV resume after a 5-minute GPU eviction window")
    figure.tight_layout()
    output = args.output_dir / "kv_pause_strategies.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
