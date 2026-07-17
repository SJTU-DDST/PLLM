from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pllm.cost_model import CalibrationProfile


def percentile(values: list[float], fraction: float = 0.95) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload if isinstance(payload, list) else [payload])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PLLM p95 cost profile")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".local" / "share" / "pllm" / "calibration.json",
    )
    parser.add_argument("--foreground-duration-seconds", type=float, default=300.0)
    args = parser.parse_args()
    rows = load_rows(args.inputs)
    yields: list[float] = []
    hibernates: list[float] = []
    local_restores: list[float] = []
    remote_restores: list[float] = []
    reclaim_ratios: list[float] = []

    for row in rows:
        level = int(row.get("level", -1))
        sleep_ms = float(row.get("sleep_seconds", 0.0)) * 1000
        if sleep_ms > 0:
            (yields if level == 0 else hibernates).append(sleep_ms)
        wake_ms = float(row.get("wake_seconds", 0.0)) * 1000
        if wake_ms > 0:
            local_restores.append(wake_ms)
        remote_ms = float(row.get("remote_restore_seconds", 0.0)) * 1000
        if remote_ms > 0:
            remote_restores.append(remote_ms)
        before = row.get("gpu_used_before_gb")
        reclaimed = row.get("reclaimed_gb")
        if isinstance(before, (int, float)) and before > 0 and isinstance(reclaimed, (int, float)):
            reclaim_ratios.append(max(0.0, min(1.0, reclaimed / before)))

    defaults = CalibrationProfile()
    profile = CalibrationProfile(
        yield_latency_ms=percentile(yields) or defaults.yield_latency_ms,
        hibernate_latency_ms=percentile(hibernates) or defaults.hibernate_latency_ms,
        local_restore_ms=percentile(local_restores) or defaults.local_restore_ms,
        remote_restore_ms=percentile(remote_restores) or defaults.remote_restore_ms,
        hibernate_reclaim_ratio=(
            sum(reclaim_ratios) / len(reclaim_ratios)
            if reclaim_ratios
            else defaults.hibernate_reclaim_ratio
        ),
        foreground_duration_seconds=args.foreground_duration_seconds,
        sample_count=len(rows),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(profile.to_dict(), indent=2))


if __name__ == "__main__":
    main()
