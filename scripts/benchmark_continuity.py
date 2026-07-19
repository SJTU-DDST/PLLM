from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests


def generated_text(payload: dict[str, Any]) -> str:
    return "".join(str(payload.get(key) or "") for key in ("reasoning", "content"))


def completion(base: str, payload: dict[str, Any]) -> str:
    response = requests.post(
        f"{base}/v1/chat/completions", json=payload, timeout=(10, 1800)
    )
    response.raise_for_status()
    return generated_text(response.json()["choices"][0]["message"])


def post(base: str, path: str, **kwargs) -> None:
    response = requests.post(f"{base}{path}", timeout=(10, 1800), **kwargs)
    response.raise_for_status()


def directory_size(root: Path | None) -> tuple[int, int]:
    if root is None or not root.exists():
        return 0, 0
    files = 0
    size = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        size += path.stat().st_size
    return files, size


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate same-stream vLLM continuity")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="nvidia/nemotron-3-super")
    parser.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--pause-after-chunks", type=int, default=8)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-repeat", type=int, default=0)
    parser.add_argument("--direct-deep", action="store_true")
    parser.add_argument("--hibercache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/continuity.json"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    prompt = "Write a deterministic Python radix sort and explain its invariants."
    if args.prompt_repeat:
        prompt += "\n" + "\n".join(
            f"Reference line {index}: stable radix passes preserve equal-key order."
            for index in range(args.prompt_repeat)
        )
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "seed": 1234,
    }
    baseline = completion(base, payload)
    stream_payload = {**payload, "stream": True}
    chunks: list[str] = []
    chunk_times: list[float] = []
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
                    delta = event["choices"][0].get("delta", {})
                    content = generated_text(delta)
                    if content:
                        chunks.append(str(content))
                        chunk_times.append(time.perf_counter())
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

    cache_files_before, cache_bytes_before = directory_size(args.hibercache_dir)
    pause_started = time.perf_counter()
    level_zero_seconds = 0.0
    if args.level > 0 and not args.direct_deep:
        level_zero_started = time.perf_counter()
        post(base, "/sleep", params={"level": 0, "mode": "keep"})
        level_zero_seconds = time.perf_counter() - level_zero_started
    sleep_started = time.perf_counter()
    post(base, "/sleep", params={"level": args.level, "mode": "keep"})
    sleep_seconds = time.perf_counter() - sleep_started
    frozen_chunk_count = len(chunks)
    frozen_chunk_time_count = len(chunk_times)
    hold_started = time.perf_counter()
    time.sleep(args.pause_seconds)
    hold_seconds = time.perf_counter() - hold_started
    chunks_during_pause = len(chunks) - frozen_chunk_count
    wake_started = time.perf_counter()
    if args.level == 0:
        post(base, "/wake_up", params=[("tags", "scheduling")])
    elif args.level == 2:
        post(base, "/wake_up", params=[("tags", "weights")])
        post(base, "/collective_rpc", json={"method": "reload_weights"})
        post(base, "/wake_up", params=[("tags", "kv_cache")])
    else:
        post(base, "/wake_up")
    wake_seconds = time.perf_counter() - wake_started
    wake_finished = time.perf_counter()
    thread.join(timeout=1800)
    cache_files_after, cache_bytes_after = directory_size(args.hibercache_dir)
    resumed = "".join(chunks)
    first_resumed_chunk_seconds = None
    if len(chunk_times) > frozen_chunk_time_count:
        first_resumed_chunk_seconds = chunk_times[frozen_chunk_time_count] - wake_finished
    result = {
        "created_at": time.time(),
        "level": args.level,
        "mode": "keep",
        "pause_at_chunk": frozen_chunk_count,
        "chunks_during_pause": chunks_during_pause,
        "level_zero_seconds": level_zero_seconds,
        "sleep_seconds": sleep_seconds,
        "hold_seconds": hold_seconds,
        "wake_seconds": wake_seconds,
        "wake_to_first_chunk_seconds": first_resumed_chunk_seconds,
        "stream_finished": not thread.is_alive(),
        "stream_error": stream_error,
        "pause_and_restore_seconds": time.perf_counter() - pause_started,
        "resume_critical_path_seconds": (
            wake_seconds + (first_resumed_chunk_seconds or 0.0)
        ),
        "prompt_characters": len(prompt),
        "hibercache_files_before": cache_files_before,
        "hibercache_files_after": cache_files_after,
        "hibercache_bytes_before": cache_bytes_before,
        "hibercache_bytes_after": cache_bytes_after,
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
