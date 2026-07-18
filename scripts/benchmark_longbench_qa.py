#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODEL = "nvidia/nemotron-3-super"
DEFAULT_MODEL_PATH = Path(
    "/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
)
DEFAULT_EXPERT_PATH = Path("/mnt/ssd-storage/cong/pllm-experts")
DEFAULT_HIBERCACHE_PATH = Path("/mnt/ssd-storage/pllm-cache")
DEFAULT_DATASETS = (
    ("mqa", Path("test_data/mqa.jsonl"), 64),
    ("nqa", Path("test_data/nqa.jsonl"), 128),
    ("tqa", Path("test_data/tqa.jsonl"), 32),
)


def normalize_answer(value: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punctuation(text: str) -> str:
        return "".join(character for character in text if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(value.lower())).split())


def qa_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    matches = sum(common.values())
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    if matches == 0:
        return 0.0
    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, references: list[str]) -> float:
    return max((qa_f1(prediction, reference) for reference in references), default=0.0)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def directory_bytes(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    files = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
                files += 1
            except FileNotFoundError:
                continue
    return total, files


def storage_snapshot(
    model_path: Path,
    expert_path: Path,
    hibercache_path: Path,
) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for name, path in (
        ("shared_model", model_path),
        ("eer_runtime_experts", expert_path),
        ("hibercache", hibercache_path),
    ):
        used_bytes, files = directory_bytes(path)
        paths[name] = {
            "path": str(path),
            "bytes": used_bytes,
            "gib": used_bytes / (1024**3),
            "files": files,
        }
    return {"captured_at": time.time(), "paths": paths}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GpuSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, float]] = []
        self.error = ""

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self._samples:
            return {"samples": 0, "error": self.error or "no samples"}

        def values(key: str) -> list[float]:
            return [sample[key] for sample in self._samples]

        timestamps = values("timestamp")
        power = values("power_w")
        energy_wh = 0.0
        for previous, current in zip(self._samples, self._samples[1:]):
            delta = current["timestamp"] - previous["timestamp"]
            energy_wh += (previous["power_w"] + current["power_w"]) * 0.5 * delta / 3600
        return {
            "samples": len(self._samples),
            "duration_seconds": max(timestamps) - min(timestamps),
            "gpu_memory_mib_min": min(values("memory_mib")),
            "gpu_memory_mib_mean": sum(values("memory_mib")) / len(self._samples),
            "gpu_memory_mib_peak": max(values("memory_mib")),
            "gpu_util_percent_mean": sum(values("gpu_util")) / len(self._samples),
            "gpu_util_percent_peak": max(values("gpu_util")),
            "power_w_mean": sum(power) / len(power),
            "power_w_peak": max(power),
            "energy_wh_estimate": energy_wh,
            "error": self.error,
        }

    def _run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop.is_set():
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._samples.append(
                    {
                        "timestamp": time.monotonic(),
                        "memory_mib": memory.used / (1024**2),
                        "gpu_util": float(utilization.gpu),
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    }
                )
                self._stop.wait(self.interval_seconds)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.error = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    max_tokens: int


def parse_dataset_spec(value: str) -> DatasetSpec:
    try:
        name, path, max_tokens = value.split("=", maxsplit=2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH=MAX_TOKENS") from exc
    if not name or int(max_tokens) <= 0:
        raise argparse.ArgumentTypeError("dataset name and max tokens must be positive")
    return DatasetSpec(name=name, path=Path(path), max_tokens=int(max_tokens))


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            records.append(json.loads(line))
    if len(records) < limit:
        raise ValueError(f"{path} has {len(records)} rows, expected at least {limit}")
    return records


def completed_results(path: Path) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            results[int(record["sample_index"])] = record
    return results


def run_request(
    *,
    base_url: str,
    model: str,
    dataset: str,
    sample_index: int,
    record: dict[str, Any],
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    references = [str(answer) for answer in record.get("answers", [])]
    payload = {
        "model": model,
        "prompt": record["prompt"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
    }
    started = time.perf_counter()
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/v1/completions",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            prediction = str(choice.get("text", "")).strip()
            usage = body.get("usage") or {}
            return {
                "schema_version": 1,
                "dataset": dataset,
                "sample_index": sample_index,
                "sample_id": record.get("_id", f"{dataset}-{sample_index}"),
                "length_field": record.get("length"),
                "references": references,
                "prediction": prediction,
                "f1": best_qa_f1(prediction, references),
                "latency_seconds": time.perf_counter() - started,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "finish_reason": choice.get("finish_reason"),
                "attempts": attempt + 1,
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 5))
    return {
        "schema_version": 1,
        "dataset": dataset,
        "sample_index": sample_index,
        "sample_id": record.get("_id", f"{dataset}-{sample_index}"),
        "length_field": record.get("length"),
        "references": references,
        "prediction": "",
        "f1": 0.0,
        "latency_seconds": time.perf_counter() - started,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "finish_reason": None,
        "attempts": retries + 1,
        "status": "error",
        "error": last_error,
    }


def summarize_dataset(
    *,
    spec: DatasetSpec,
    records: list[dict[str, Any]],
    results: dict[int, dict[str, Any]],
    wall_seconds: float,
    gpu: dict[str, Any],
) -> dict[str, Any]:
    selected = [results[index] for index in range(len(records)) if index in results]
    successful = [result for result in selected if result["status"] == "ok"]
    latencies = [float(result["latency_seconds"]) for result in successful]
    prompt_tokens = sum(int(result["prompt_tokens"]) for result in successful)
    completion_tokens = sum(int(result["completion_tokens"]) for result in successful)
    return {
        "dataset": spec.name,
        "path": str(spec.path),
        "source_sha256": file_sha256(spec.path),
        "requested_samples": len(records),
        "completed_samples": len(successful),
        "errors": len(selected) - len(successful),
        "max_tokens": spec.max_tokens,
        "wall_seconds": wall_seconds,
        "f1": sum(float(result["f1"]) for result in successful) / len(successful)
        if successful
        else 0.0,
        "latency_seconds_mean": sum(latencies) / len(latencies) if latencies else None,
        "latency_seconds_p50": percentile(latencies, 0.50),
        "latency_seconds_p95": percentile(latencies, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_per_second": prompt_tokens / wall_seconds if wall_seconds else 0.0,
        "completion_tokens_per_second": completion_tokens / wall_seconds if wall_seconds else 0.0,
        "total_tokens_per_second": (prompt_tokens + completion_tokens) / wall_seconds
        if wall_seconds
        else 0.0,
        "samples_per_second": len(successful) / wall_seconds if wall_seconds else 0.0,
        "gpu": gpu,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the first N LongBench QA rows")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qa_benchmark"))
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--expert-path", type=Path, default=DEFAULT_EXPERT_PATH)
    parser.add_argument("--hibercache-path", type=Path, default=DEFAULT_HIBERCACHE_PATH)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.concurrency <= 0 or args.timeout <= 0 or args.retries < 0:
        parser.error("limit, concurrency and timeout must be positive; retries cannot be negative")

    specs = args.dataset or [DatasetSpec(*dataset) for dataset in DEFAULT_DATASETS]
    mode_dir = args.output_dir / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    summary_path = mode_dir / "summary.json"
    prior_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() and not args.overwrite
        else {}
    )
    storage_before = storage_snapshot(args.model_path, args.expert_path, args.hibercache_path)
    started_at = prior_summary.get("started_at", datetime.now().astimezone().isoformat())
    dataset_summaries: dict[str, Any] = {}

    for spec in specs:
        records = load_records(spec.path, args.limit)
        output_path = mode_dir / f"{spec.name}.jsonl"
        if args.overwrite and output_path.exists():
            output_path.unlink()
        existing = completed_results(output_path)
        pending = [index for index in range(len(records)) if existing.get(index, {}).get("status") != "ok"]
        lock = threading.Lock()
        sampler = GpuSampler()
        sampler.start()
        wall_started = time.perf_counter()
        with output_path.open("a", encoding="utf-8", buffering=1) as output:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        run_request,
                        base_url=args.base_url,
                        model=args.model,
                        dataset=spec.name,
                        sample_index=index,
                        record=records[index],
                        max_tokens=spec.max_tokens,
                        timeout_seconds=args.timeout,
                        retries=args.retries,
                    ): index
                    for index in pending
                }
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    with lock:
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    existing[int(result["sample_index"])] = result
                    completed += 1
                    print(
                        f"[{args.mode}/{spec.name}] {completed}/{len(pending)} "
                        f"index={result['sample_index']} status={result['status']} "
                        f"latency={result['latency_seconds']:.3f}s f1={result['f1']:.4f}",
                        flush=True,
                    )
        previous_wall_seconds = float(
            prior_summary.get("datasets", {}).get(spec.name, {}).get("wall_seconds", 0.0)
        )
        wall_seconds = previous_wall_seconds + time.perf_counter() - wall_started
        gpu = sampler.stop()
        dataset_summaries[spec.name] = summarize_dataset(
            spec=spec,
            records=records,
            results=existing,
            wall_seconds=wall_seconds,
            gpu=gpu,
        )
        write_json(
            summary_path,
            {
                "schema_version": 1,
                "mode": args.mode,
                "started_at": started_at,
                "updated_at": datetime.now().astimezone().isoformat(),
                "base_url": args.base_url,
                "model": args.model,
                "limit": args.limit,
                "concurrency": args.concurrency,
                "metadata": json.loads(args.metadata_json),
                "storage_before": storage_before,
                "storage_after": storage_snapshot(
                    args.model_path, args.expert_path, args.hibercache_path
                ),
                "datasets": dataset_summaries,
            },
        )

    successful = sum(item["completed_samples"] for item in dataset_summaries.values())
    weighted_f1 = (
        sum(item["f1"] * item["completed_samples"] for item in dataset_summaries.values())
        / successful
        if successful
        else 0.0
    )
    total_wall = sum(item["wall_seconds"] for item in dataset_summaries.values())
    total_prompt = sum(item["prompt_tokens"] for item in dataset_summaries.values())
    total_completion = sum(item["completion_tokens"] for item in dataset_summaries.values())
    final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_summary["finished_at"] = datetime.now().astimezone().isoformat()
    final_summary["aggregate"] = {
        "completed_samples": successful,
        "errors": sum(item["errors"] for item in dataset_summaries.values()),
        "macro_f1": sum(item["f1"] for item in dataset_summaries.values())
        / len(dataset_summaries),
        "sample_weighted_f1": weighted_f1,
        "dataset_wall_seconds": total_wall,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "prompt_tokens_per_second": total_prompt / total_wall if total_wall else 0.0,
        "completion_tokens_per_second": total_completion / total_wall if total_wall else 0.0,
        "total_tokens_per_second": (total_prompt + total_completion) / total_wall
        if total_wall
        else 0.0,
    }
    write_json(summary_path, final_summary)
    print(json.dumps(final_summary["aggregate"], indent=2), flush=True)
    return 0 if final_summary["aggregate"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
