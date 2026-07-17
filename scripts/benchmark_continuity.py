from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests


def completion(base: str, payload: dict[str, Any]) -> str:
    response = requests.post(
        f"{base}/v1/chat/completions", json=payload, timeout=(10, 1800)
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"].get("content", ""))


def post(base: str, path: str, **kwargs) -> None:
    response = requests.post(f"{base}{path}", timeout=(10, 1800), **kwargs)
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate same-stream vLLM continuity")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="nvidia/nemotron-3-super")
    parser.add_argument("--level", type=int, choices=(0, 2), default=0)
    parser.add_argument("--pause-after-chunks", type=int, default=8)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/continuity.json"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Write a deterministic Python radix sort and explain its invariants.",
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "seed": 1234,
    }
    baseline = completion(base, payload)
    stream_payload = {**payload, "stream": True}
    chunks: list[str] = []
    pause_ready = threading.Event()
    stream_error: list[str] = []

    def consume() -> None:
        try:
            with requests.post(
                f"{base}/v1/chat/completions",
                json=stream_payload,
                stream=True,
                timeout=(10, 1800),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    content = event["choices"][0].get("delta", {}).get("content")
                    if content:
                        chunks.append(str(content))
                        if len(chunks) >= args.pause_after_chunks:
                            pause_ready.set()
        except Exception as exc:  # recorded in result for the GPU experiment
            stream_error.append(str(exc))
            pause_ready.set()

    thread = threading.Thread(target=consume)
    thread.start()
    if not pause_ready.wait(timeout=300):
        raise SystemExit("stream did not reach the pause boundary")
    if stream_error:
        raise SystemExit(stream_error[0])

    pause_started = time.perf_counter()
    post(base, "/sleep", params={"level": args.level, "mode": "keep"})
    frozen_chunk_count = len(chunks)
    time.sleep(args.pause_seconds)
    chunks_during_pause = len(chunks) - frozen_chunk_count
    if args.level == 0:
        post(base, "/wake_up", params=[("tags", "scheduling")])
    else:
        post(base, "/wake_up", params=[("tags", "weights")])
        post(base, "/collective_rpc", json={"method": "reload_weights"})
        post(base, "/wake_up", params=[("tags", "kv_cache")])
    thread.join(timeout=1800)
    resumed = "".join(chunks)
    result = {
        "created_at": time.time(),
        "level": args.level,
        "mode": "keep",
        "pause_at_chunk": frozen_chunk_count,
        "chunks_during_pause": chunks_during_pause,
        "stream_finished": not thread.is_alive(),
        "stream_error": stream_error,
        "pause_and_restore_seconds": time.perf_counter() - pause_started,
        "baseline_characters": len(baseline),
        "resumed_characters": len(resumed),
        "exact_match": resumed == baseline,
        "baseline_sha256": __import__("hashlib").sha256(baseline.encode()).hexdigest(),
        "resumed_sha256": __import__("hashlib").sha256(resumed.encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["exact_match"] or chunks_during_pause:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
