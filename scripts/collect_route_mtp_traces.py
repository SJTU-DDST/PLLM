#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from pllm.config import pllm_runtime_dir
from pllm.vllm_eer_runtime import request_runtime


DEFAULT_DATASETS = (
    ("mqa", Path("test_data/mqa.jsonl")),
    ("nqa", Path("test_data/nqa.jsonl")),
    ("tqa", Path("test_data/tqa.jsonl")),
)


def extract_question(prompt: str) -> str:
    multifield = re.search(
        r"The question is as follows:\s*(.*?)\s*The context is as follows:",
        prompt,
        flags=re.DOTALL,
    )
    if multifield is not None:
        return multifield.group(1).strip()
    questions = re.findall(
        r"(?:^|\n)Question:\s*(.*?)\s*(?:\nAnswer:|\Z)",
        prompt,
        flags=re.DOTALL,
    )
    if questions:
        return questions[-1].strip()
    raise ValueError("prompt does not contain a recognized question boundary")


def load_prompts(
    path: Path,
    limit: int,
    max_source_tokens: int,
    *,
    question_only: bool,
) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            if (
                not question_only
                and int(record.get("length", 0)) > max_source_tokens
            ):
                continue
            prompt = str(record.get("prompt", ""))
            if not prompt:
                continue
            if question_only:
                try:
                    question = extract_question(prompt)
                except ValueError:
                    continue
                prompt = (
                    "Answer this question briefly and accurately. "
                    f"Question: {question}"
                )
            records.append({"index": index, "prompt": prompt, "length": record.get("length")})
            if len(records) >= limit:
                break
    return records


def wait_for_shadow(socket: Path) -> dict[str, Any]:
    status = None
    for _ in range(600):
        status = request_runtime(socket, {"command": "status"}, timeout=30)
        route = status.get("route_mtp", {})
        if int(route.get("queue_depth", 0)) == 0:
            return status
        time.sleep(0.05)
    raise TimeoutError("RouteMTP worker did not drain within 30 seconds")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect request-separated MTP features and next-token target routes"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="nvidia/nemotron-3-super")
    parser.add_argument("--socket", type=Path, default=pllm_runtime_dir() / "pllm-eer.sock")
    parser.add_argument("--per-dataset", type=int, default=6)
    parser.add_argument("--max-source-tokens", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--question-only",
        action="store_true",
        help="extract the final question to keep partial-resident prefill exact",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.per_dataset < 1 or args.max_tokens < 2:
        parser.error("per-dataset must be positive and max-tokens must be at least 2")

    entries = []
    started = time.time()
    for dataset, path in DEFAULT_DATASETS:
        prompts = load_prompts(
            path,
            args.per_dataset,
            args.max_source_tokens,
            question_only=args.question_only,
        )
        if len(prompts) < args.per_dataset:
            raise RuntimeError(f"{path} supplied only {len(prompts)} eligible prompts")
        for position, record in enumerate(prompts, 1):
            request_runtime(
                args.socket,
                {"command": "phase", "phase": "decode", "reset_decode": True},
                timeout=30,
            )
            request_started = time.perf_counter()
            response = requests.post(
                f"{args.base_url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": record["prompt"]}],
                    "temperature": 0.0,
                    "max_tokens": args.max_tokens,
                },
                timeout=900,
            )
            response.raise_for_status()
            payload = response.json()
            status = wait_for_shadow(args.socket)
            flushed = request_runtime(
                args.socket,
                {"command": "route_mtp_trace_flush"},
                timeout=120,
            )
            usage = payload.get("usage", {})
            entry = {
                "dataset": dataset,
                "dataset_index": record["index"],
                "source_length": record["length"],
                "prompt_mode": "question_only" if args.question_only else "full",
                "position": position,
                "wall_seconds": time.perf_counter() - request_started,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "finish_reason": payload["choices"][0].get("finish_reason"),
                "trace_files": flushed["files"],
                "trace_samples": flushed["samples"],
                "gpu_probe": status["route_mtp"]["gpu_probe"],
            }
            entries.append(entry)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "started_at_unix": started,
                        "updated_at_unix": time.time(),
                        "entries": entries,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[{dataset}] {position}/{len(prompts)} tokens="
                f"{entry['completion_tokens']} paired={entry['trace_samples']} "
                f"wall={entry['wall_seconds']:.2f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
