from __future__ import annotations

import argparse
import json
from pathlib import Path


COLORS = ["#37b878", "#58a6ff", "#f0b64d", "#8b99a8"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PLLM benchmark JSON as SVG")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/sleep_comparison.svg"))
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    values = [float(row.get("reclaimed_gb") or 0) for row in rows]
    maximum = max(values + [1.0])
    width, height = 760, 320
    chart_left, chart_bottom, chart_height = 90, 260, 190
    bar_width = 90
    gap = 65
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#20272c"/>',
        '<text x="32" y="38" fill="#f2f5f6" font-size="20" font-family="sans-serif">PLLM GPU Memory Reclaimed</text>',
        f'<line x1="{chart_left}" y1="{chart_bottom}" x2="720" y2="{chart_bottom}" stroke="#69777f"/>',
    ]
    for index, (row, value) in enumerate(zip(rows, values)):
        x = chart_left + 45 + index * (bar_width + gap)
        bar_height = chart_height * value / maximum
        y = chart_bottom - bar_height
        label = f"L{row.get('level', '?')}"
        parts.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="4" fill="{COLORS[index % len(COLORS)]}"/>',
                f'<text x="{x + bar_width / 2}" y="{y - 8:.1f}" text-anchor="middle" fill="#dfe6e8" font-size="14" font-family="sans-serif">{value:.1f} GiB</text>',
                f'<text x="{x + bar_width / 2}" y="{chart_bottom + 24}" text-anchor="middle" fill="#b8c3c8" font-size="13" font-family="sans-serif">{label}</text>',
            ]
        )
    parts.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()

