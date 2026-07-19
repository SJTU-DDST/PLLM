from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot live pause/resume critical paths")
    parser.add_argument("inputs", nargs="+", metavar="LABEL=JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for item in args.inputs:
        label, separator, filename = item.partition("=")
        if not separator:
            parser.error(f"input must be LABEL=JSON: {item}")
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        entry = float(payload.get("level_zero_seconds", 0.0)) + float(
            payload.get("sleep_seconds", 0.0)
        )
        wake = float(payload.get("wake_seconds", 0.0))
        first = float(payload.get("wake_to_first_chunk_seconds") or 0.0)
        rows.append(
            {
                "label": label,
                "level": int(payload["level"]),
                "entry_seconds": entry,
                "wake_api_seconds": wake,
                "wake_to_first_chunk_seconds": first,
                "critical_seconds": entry + wake + first,
                "hold_seconds": float(payload["hold_seconds"]),
                "continuous": bool(
                    payload.get("stream_finished")
                    and not payload.get("stream_error")
                    and payload.get("chunks_during_pause") == 0
                ),
                "cross_request_exact_match": bool(payload.get("exact_match")),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "live_pause_resume_e2e.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    entry = [row["entry_seconds"] for row in rows]
    wake = [row["wake_api_seconds"] for row in rows]
    first = [row["wake_to_first_chunk_seconds"] for row in rows]
    figure, axis = plt.subplots(figsize=(8.6, 4.9))
    axis.bar(x, entry, label="Pause entry", color="#5b8ff9")
    axis.bar(x, wake, bottom=entry, label="Wake API", color="#61d9a3")
    bottom = [a + b for a, b in zip(entry, wake)]
    axis.bar(x, first, bottom=bottom, label="Wake to first chunk", color="#d97757")
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Seconds (pause hold excluded)")
    axis.set_title("Live EER pause/resume critical path")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    for index, row in enumerate(rows):
        axis.text(
            index,
            row["critical_seconds"],
            f"{row['critical_seconds']:.1f}s",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    output = args.output_dir / "live_pause_resume_e2e.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
