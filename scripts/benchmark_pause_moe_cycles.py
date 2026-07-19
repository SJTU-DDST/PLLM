#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pllm.config import pllm_runtime_dir
from pllm.expert_runtime_client import ExpertRuntimeClient


DEFAULT_PROMPT = (
    "Output the integers from 1 through 96 in order, separated by single spaces. "
    "Do not add any explanation."
)
COUNTER_KEYS = (
    "hits",
    "misses",
    "evictions",
    "bytes_loaded",
    "load_time_ns",
    "resize_count",
    "resize_time_ns",
    "gpu_copy_bytes",
    "batch_loads",
    "batch_objects",
    "capacity_change_count",
    "capacity_change_time_ns",
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def interval_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values) * 1000 if values else None,
        "p50_ms": percentile(values, 0.50) * 1000 if values else None,
        "p95_ms": percentile(values, 0.95) * 1000 if values else None,
    }


def gpu_memory_mib() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    except Exception:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={os.environ.get('PLLM_GPU_INDEX', '0')}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(result.stdout.strip().splitlines()[0])
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            return None


def runtime_status(
    session: requests.Session,
    base_url: str,
    runtime: ExpertRuntimeClient | None = None,
) -> dict[str, Any]:
    if runtime is not None:
        payload = runtime.request({"command": "status"}, timeout=10)
        if not payload.get("data_plane_ready"):
            raise RuntimeError(f"expert data plane is not ready: {payload}")
        return payload
    response = session.get(
        f"{base_url.rstrip('/')}/api/v1/expert-dataplane", timeout=10
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("data_plane_ready"):
        raise RuntimeError(f"expert data plane is not ready: {payload}")
    return payload


def aggregate_counters(status: dict[str, Any]) -> dict[str, int]:
    totals = {key: 0 for key in COUNTER_KEYS}
    for layer in status.get("data_plane", {}).get("layers", []):
        counters = layer.get("counters", {})
        for key in COUNTER_KEYS:
            totals[key] += int(counters.get(key, 0))
    return totals


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in COUNTER_KEYS}


def resize(
    session: requests.Session,
    base_url: str,
    slots: int,
    retain_policy: str,
    timeout: float,
    runtime: ExpertRuntimeClient | None = None,
    residency_mode: str = "logical",
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if runtime is not None:
        paused = session.post(
            f"{base_url.rstrip('/')}/sleep",
            params={"level": 0, "mode": "keep"},
            timeout=timeout,
        )
        paused.raise_for_status()
        try:
            if residency_mode == "physical":
                result = runtime.resize(slots, retain_policy=retain_policy)
            else:
                result = runtime.set_capacity(slots, retain_policy=retain_policy)
        except Exception:
            raise
        finally:
            resumed = session.post(
                f"{base_url.rstrip('/')}/wake_up",
                params=[("tags", "scheduling")],
                timeout=timeout,
            )
            resumed.raise_for_status()
        return result, time.perf_counter() - started
    response = session.post(
        f"{base_url.rstrip('/')}/api/v1/expert-dataplane/actions",
        json={
            "action": "resize",
            "slots_per_layer": slots,
            "retain_policy": retain_policy,
            "phase": "decode" if retain_policy == "decode_hot" else "idle",
        },
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.json(), elapsed


def pause_wake(
    session: requests.Session,
    base_url: str,
    timeout: float,
    direct: bool = False,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if direct:
        paused = session.post(
            f"{base_url.rstrip('/')}/sleep",
            params={"level": 0, "mode": "keep"},
            timeout=timeout,
        )
        paused.raise_for_status()
        resumed = session.post(
            f"{base_url.rstrip('/')}/wake_up",
            params=[("tags", "scheduling")],
            timeout=timeout,
        )
        resumed.raise_for_status()
        return {
            "pause_status": paused.status_code,
            "wake_status": resumed.status_code,
        }, time.perf_counter() - started
    paused = session.post(
        f"{base_url.rstrip('/')}/api/v1/actions",
        json={"action": "yield"},
        timeout=timeout,
    )
    paused.raise_for_status()
    resumed = session.post(
        f"{base_url.rstrip('/')}/api/v1/actions",
        json={"action": "wake"},
        timeout=timeout,
    )
    resumed.raise_for_status()
    return {
        "pause": paused.json(),
        "wake": resumed.json(),
    }, time.perf_counter() - started


def stream_round(
    *,
    session: requests.Session,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    arm: str,
    action_mode: str,
    target_slots: int,
    trigger_events: int,
    action_timeout: float,
    baseline_slots: int,
    runtime: ExpertRuntimeClient | None,
    residency_mode: str,
) -> dict[str, Any]:
    if runtime is not None:
        runtime.set_phase("prefill", reset_decode=True)
    before = runtime_status(session, base_url, runtime)
    if int(before.get("slots_per_layer", 0)) != baseline_slots:
        resize(
            session,
            base_url,
            baseline_slots,
            "lru",
            action_timeout,
            runtime,
            residency_mode,
        )
        before = runtime_status(session, base_url, runtime)
    counters_before = aggregate_counters(before)
    memory_before_mib = gpu_memory_mib()
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
    }
    started = time.perf_counter()
    response = session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=request,
        stream=True,
        timeout=(10, None),
    )
    response.raise_for_status()
    header_seconds = time.perf_counter() - started
    event_times: list[float] = []
    captured: list[str] = []
    transition: dict[str, Any] | None = None
    transition_event_index: int | None = None

    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        if line == b"data: [DONE]":
            break
        event = json.loads(line[6:])
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        text = (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
        )
        if not text:
            continue
        captured.append(str(text))
        event_times.append(time.perf_counter())
        if runtime is not None and len(event_times) == 1:
            runtime.set_phase("decode")
        if action_mode != "none" and transition is None and len(event_times) >= trigger_events:
            memory_pre_resize_mib = gpu_memory_mib()
            if action_mode == "pause_only":
                result, action_seconds = pause_wake(
                    session,
                    base_url,
                    action_timeout,
                    direct=runtime is not None,
                )
            else:
                result, action_seconds = resize(
                    session,
                    base_url,
                    target_slots,
                    "decode_hot",
                action_timeout,
                runtime,
                residency_mode,
                )
            transition_event_index = len(event_times) - 1
            transition = {
                "action": action_mode,
                "target_slots": target_slots,
                "action_seconds": action_seconds,
                "memory_pre_resize_mib": memory_pre_resize_mib,
                "memory_post_resize_mib": gpu_memory_mib(),
                "response": result,
            }
    response.close()
    if runtime is not None:
        runtime.set_phase("idle")
    finished = time.perf_counter()
    after = runtime_status(session, base_url, runtime)
    counters_after = aggregate_counters(after)
    intervals = [current - previous for previous, current in zip(event_times, event_times[1:])]
    before_intervals = intervals
    after_intervals: list[float] = []
    transition_gap = None
    if transition_event_index is not None:
        before_intervals = intervals[:transition_event_index]
        if transition_event_index < len(intervals):
            transition_gap = intervals[transition_event_index]
            after_intervals = intervals[transition_event_index + 1 :]
    return {
        "status": "ok",
        "arm": arm,
        "target_slots": target_slots,
        "wall_seconds": finished - started,
        "response_header_seconds": header_seconds,
        "ttft_seconds": event_times[0] - started if event_times else None,
        "text_events": len(event_times),
        "output": "".join(captured),
        "all_tpot": interval_summary(intervals),
        "pre_transition_tpot": interval_summary(before_intervals),
        "post_transition_tpot": interval_summary(after_intervals),
        "transition_gap_ms": transition_gap * 1000 if transition_gap is not None else None,
        "transition": transition,
        "memory_before_request_mib": memory_before_mib,
        "memory_after_request_mib": gpu_memory_mib(),
        "counter_delta": counter_delta(counters_before, counters_after),
        "runtime_after": {
            "slots_per_layer": after.get("slots_per_layer"),
            "route_trace": after.get("route_trace"),
            "state_island": after.get("state_island"),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeatedly pause, shrink, and restore live MoE decode residency"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:17861")
    parser.add_argument(
        "--control-mode", choices=("daemon", "direct"), default="daemon"
    )
    parser.add_argument("--runtime-socket", type=Path)
    parser.add_argument(
        "--residency-mode", choices=("logical", "physical"), default="logical"
    )
    parser.add_argument("--baseline-slots", type=int, default=512)
    parser.add_argument("--model", default="nvidia/nemotron-3-super")
    parser.add_argument("--profiles", default="504,496,480")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--trigger-events", type=int, default=8)
    parser.add_argument("--action-timeout", type=float, default=900)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = sorted(
        {int(item) for item in args.profiles.split(",") if item.strip()}
    )
    if (
        args.rounds <= 0
        or args.warmup_rounds < 0
        or args.max_tokens <= args.trigger_events
        or args.trigger_events <= 0
        or args.action_timeout <= 0
        or args.baseline_slots < 22
        or args.baseline_slots > 512
        or any(profile < 22 or profile >= args.baseline_slots for profile in profiles)
    ):
        parser.error("invalid rounds, token counts, timeout, or slot profile")

    session = requests.Session()
    runtime = None
    if args.control_mode == "direct":
        socket_path = args.runtime_socket or pllm_runtime_dir() / "pllm-eer.sock"
        runtime = ExpertRuntimeClient(str(socket_path), timeout_seconds=args.action_timeout)
    elif args.residency_mode == "logical":
        parser.error("logical residency mode requires --control-mode direct")
    initial_status = runtime_status(session, args.base_url, runtime)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence": (
            "live_token_boundary_level0_pause_logical_capacity_wake"
            if args.residency_mode == "logical"
            else "live_token_boundary_level0_pause_physical_resize_wake"
        ),
        "base_url": args.base_url,
        "model": args.model,
        "arms": ["resident_baseline", "pause_only", *[f"pause_evict_{value}" for value in profiles]],
        "control_mode": args.control_mode,
        "residency_mode": args.residency_mode,
        "runtime_profile": {
            "physical_slots_per_layer": initial_status.get(
                "physical_slots_per_layer"
            ),
            "recent_pin_steps": initial_status.get("recent_pin_steps", 0),
        },
        "baseline_slots": args.baseline_slots,
        "profiles": [args.baseline_slots, *profiles],
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "max_tokens": args.max_tokens,
        "trigger_events": args.trigger_events,
        "results": [],
    }
    write_json(args.output, payload)
    try:
        for warmup_index in range(args.warmup_rounds):
            print(
                f"warmup={warmup_index + 1}/{args.warmup_rounds}", flush=True
            )
            warmup = stream_round(
                session=session,
                base_url=args.base_url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                arm="warmup",
                action_mode="none",
                target_slots=args.baseline_slots,
                trigger_events=args.trigger_events,
                action_timeout=args.action_timeout,
                baseline_slots=args.baseline_slots,
                runtime=runtime,
                residency_mode=args.residency_mode,
            )
            if not warmup.get("output"):
                raise RuntimeError("warmup request produced no streamed text")
        for round_index in range(args.rounds):
            arms = [
                ("resident_baseline", "none", args.baseline_slots),
                ("pause_only", "pause_only", args.baseline_slots),
                *[
                    (f"pause_evict_{target}", "resize", target)
                    for target in profiles
                ],
            ]
            offset = round_index % len(arms)
            arms = arms[offset:] + arms[:offset]
            for arm, action_mode, target_slots in arms:
                print(
                    f"round={round_index + 1}/{args.rounds} arm={arm}",
                    flush=True,
                )
                try:
                    row = stream_round(
                        session=session,
                        base_url=args.base_url,
                        model=args.model,
                        prompt=args.prompt,
                        max_tokens=args.max_tokens,
                        arm=arm,
                        action_mode=action_mode,
                        target_slots=target_slots,
                        trigger_events=args.trigger_events,
                        action_timeout=args.action_timeout,
                        baseline_slots=args.baseline_slots,
                        runtime=runtime,
                        residency_mode=args.residency_mode,
                    )
                except Exception as exc:
                    row = {
                        "status": "error",
                        "arm": arm,
                        "target_slots": target_slots,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                row["round"] = round_index + 1
                payload["results"].append(row)
                write_json(args.output, payload)
                try:
                    if runtime is not None:
                        runtime.set_phase("idle")
                    status = runtime_status(session, args.base_url, runtime)
                    if int(status.get("slots_per_layer", 0)) != args.baseline_slots:
                        resize(
                            session,
                            args.base_url,
                            args.baseline_slots,
                            "lru",
                            args.action_timeout,
                            runtime,
                            args.residency_mode,
                        )
                except Exception as exc:
                    row["restore_error"] = f"{type(exc).__name__}: {exc}"
                    write_json(args.output, payload)
                    raise
    finally:
        payload["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(args.output, payload)
    errors = sum(row.get("status") != "ok" for row in payload["results"])
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
