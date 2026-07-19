#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)
POLICY_LABELS = {"lru": "LRU", "window_lfu": "Window LFU"}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def load_profiles(summary: dict[str, Any]) -> list[dict[str, Any]]:
    simulations = summary.get("aggregate", {}).get("simulations", {})
    profiles = []
    for value in simulations.values():
        if not isinstance(value, dict) or not value.get("samples"):
            continue
        row = dict(value)
        row["slots_per_layer"] = int(row["slots_per_layer"])
        profiles.append(row)
    if not profiles:
        raise ValueError("summary has no successful aggregate cache simulations")
    return profiles


def history_locality(
    route_paths: list[Path], horizons: tuple[int, ...]
) -> tuple[list[dict[str, Any]], int]:
    hits = defaultdict(int)
    accesses = defaultdict(int)
    working_set_total = defaultdict(int)
    working_set_rows = defaultdict(int)
    decode_tokens = 0

    for path in route_paths:
        with np.load(path, allow_pickle=False) as archive:
            prefill = archive["prefill_tail"]
            decode = archive["decode"]
        if decode.ndim != 3 or prefill.ndim != 3:
            raise ValueError(f"unexpected route layout in {path}")
        if decode.shape[1:] != prefill.shape[1:]:
            raise ValueError(f"prefill/decode route mismatch in {path}")
        decode_tokens += int(decode.shape[0])
        sequence = np.concatenate((prefill, decode), axis=0)
        decode_start = int(prefill.shape[0])
        for horizon in horizons:
            for token_index in range(decode_start, int(sequence.shape[0])):
                start = max(0, token_index - horizon)
                for layer in range(int(sequence.shape[1])):
                    actual = set(int(item) for item in sequence[token_index, layer])
                    history = set(
                        int(item)
                        for item in sequence[start:token_index, layer, :].reshape(-1)
                    )
                    hits[horizon] += len(actual & history)
                    accesses[horizon] += len(actual)
                    working_set_total[horizon] += len(history)
                    working_set_rows[horizon] += 1

    rows = []
    for horizon in horizons:
        rows.append(
            {
                "history_tokens": horizon,
                "expert_reuse_coverage": hits[horizon] / accesses[horizon]
                if accesses[horizon]
                else 0.0,
                "mean_history_working_set_per_layer": working_set_total[horizon]
                / working_set_rows[horizon]
                if working_set_rows[horizon]
                else 0.0,
                "expert_accesses": accesses[horizon],
            }
        )
    return rows, decode_tokens


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_hit_rate(profiles: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for policy in sorted({str(row["policy"]) for row in profiles}):
        selected = sorted(
            (row for row in profiles if row["policy"] == policy),
            key=lambda row: float(row["projected_reclaim_gib"]),
        )
        x = [float(row["projected_reclaim_gib"]) for row in selected]
        y = [100 * float(row["byte_hit_rate"]) for row in selected]
        axis.plot(x, y, marker="o", linewidth=2, label=POLICY_LABELS.get(policy, policy))
        for row, x_value, y_value in zip(selected, x, y):
            axis.annotate(
                str(row["slots_per_layer"]),
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.axhline(95, color="#b42318", linestyle="--", linewidth=1.4, label="95% guardrail")
    axis.set_xlabel("Projected model-weight reclaim (GiB)")
    axis.set_ylabel("Decode expert byte hit rate (%)")
    axis.set_title("Decode locality under exact-route cache replay")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_miss_cost(profiles: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for policy in sorted({str(row["policy"]) for row in profiles}):
        selected = sorted(
            (row for row in profiles if row["policy"] == policy),
            key=lambda row: float(row["projected_reclaim_gib"]),
        )
        x = [float(row["projected_reclaim_gib"]) for row in selected]
        label = POLICY_LABELS.get(policy, policy)
        axes[0].plot(
            x,
            [float(row["misses_per_token"]) for row in selected],
            marker="o",
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            x,
            [max(1.0, float(row["estimated_slowdown_ratio"])) for row in selected],
            marker="o",
            linewidth=2,
            label=label,
        )
    axes[0].set_ylabel("Blocking expert loads per decode token")
    axes[1].set_ylabel("Estimated TPOT slowdown (x)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Projected model-weight reclaim (GiB)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Cost of evicting decode experts")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_history_locality(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    secondary = axis.twinx()
    x = [int(row["history_tokens"]) for row in rows]
    coverage = [100 * float(row["expert_reuse_coverage"]) for row in rows]
    working_set = [float(row["mean_history_working_set_per_layer"]) for row in rows]
    first = axis.plot(x, coverage, color="#175cd3", marker="o", linewidth=2.2, label="Reuse coverage")
    second = secondary.plot(
        x,
        working_set,
        color="#067647",
        marker="s",
        linewidth=2.2,
        label="History working set",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, labels=[str(value) for value in x])
    axis.set_xlabel("Prior tokens retained per layer")
    axis.set_ylabel("Current expert accesses seen in history (%)", color="#175cd3")
    secondary.set_ylabel("Mean distinct experts in history", color="#067647")
    axis.set_title("Cross-token MoE activation locality")
    axis.grid(alpha=0.25)
    axis.legend(first + second, [item.get_label() for item in first + second], loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def choose_feasible(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [
        row
        for row in profiles
        if float(row["byte_hit_rate"]) >= 0.95
        and float(row["estimated_slowdown_ratio"]) < 5.0
    ]
    return max(
        feasible,
        key=lambda row: (
            float(row["projected_reclaim_gib"]),
            float(row["byte_hit_rate"]),
            -float(row["estimated_slowdown_ratio"]),
        ),
        default=None,
    )


def write_report(
    path: Path,
    summary: dict[str, Any],
    profiles: list[dict[str, Any]],
    locality: list[dict[str, Any]],
    route_files: int,
    decode_tokens: int,
) -> None:
    feasible = choose_feasible(profiles)
    best_history = max(locality, key=lambda row: row["history_tokens"])
    aggregate = summary.get("aggregate", {})
    lines = [
        "# MoE decode locality experiment",
        "",
        "## Evidence",
        "",
        f"- Route source: `{summary.get('evidence', 'unknown')}`",
        f"- Successful requests: {aggregate.get('completed_samples', 0)}",
        f"- Route files: {route_files}",
        f"- Captured decode tokens: {decode_tokens}",
        "- Cache replay preserves every actual Top-k route; a miss is counted as a blocking load.",
        "",
        "## Result",
        "",
    ]
    if feasible is None:
        lines.append(
            "No tested eviction profile met both the 95% byte-hit and <5x estimated-slowdown guardrails."
        )
    else:
        lines.append(
            "The most aggressive tested profile that met both guardrails was "
            f"{feasible['policy']} at {feasible['slots_per_layer']} slots/layer: "
            f"{100 * float(feasible['byte_hit_rate']):.2f}% byte hit, "
            f"{float(feasible['projected_reclaim_gib']):.2f} GiB projected reclaim, "
            f"{float(feasible['misses_per_token']):.2f} blocking loads/token, and "
            f"{float(feasible['estimated_slowdown_ratio']):.2f}x estimated TPOT."
        )
    lines.extend(
        [
            "",
            f"With {best_history['history_tokens']} prior tokens, "
            f"{100 * float(best_history['expert_reuse_coverage']):.2f}% of current expert accesses "
            "had appeared in the same layer's history, while that history contained "
            f"{float(best_history['mean_history_working_set_per_layer']):.1f} distinct experts/layer on average.",
            "",
            "## Scope",
            "",
            "The hit and reclaim results are offline replay of live full-resident routes. "
            "Latency is an estimate from configured I/O bandwidth and p95 per-object latency; "
            "it is not a live elastic-throughput measurement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot exact-route MoE decode locality results")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    horizons = tuple(sorted({int(item) for item in args.horizons.split(",")}))
    if not horizons or any(value <= 0 for value in horizons):
        parser.error("history horizons must be positive")

    summary = load_json(args.input_dir / "summary.json")
    profiles = load_profiles(summary)
    route_paths = sorted((args.input_dir / "routes").glob("**/*.npz"))
    if not route_paths:
        raise FileNotFoundError(f"no route files found below {args.input_dir / 'routes'}")
    locality, decode_tokens = history_locality(route_paths, horizons)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_fields = (
        "policy",
        "slots_per_layer",
        "samples",
        "byte_hit_rate",
        "misses_per_token",
        "estimated_miss_ms_per_token",
        "estimated_slowdown_ratio",
        "projected_reclaim_gib",
        "resident_weight_gib",
        "miss_gib",
    )
    profile_rows = [{key: row.get(key) for key in profile_fields} for row in profiles]
    write_csv(output_dir / "cache_profiles.csv", profile_rows)
    write_csv(output_dir / "history_locality.csv", locality)
    plot_hit_rate(profiles, output_dir / "hit_rate_vs_reclaim.png")
    plot_miss_cost(profiles, output_dir / "miss_cost_vs_reclaim.png")
    plot_history_locality(locality, output_dir / "token_history_locality.png")

    analysis = {
        "schema_version": 1,
        "evidence": summary.get("evidence", "unknown"),
        "route_files": len(route_paths),
        "decode_tokens": decode_tokens,
        "history_locality": locality,
        "feasible_profile": choose_feasible(profiles),
    }
    (output_dir / "locality_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output_dir / "report.md",
        summary,
        profiles,
        locality,
        len(route_paths),
        decode_tokens,
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
