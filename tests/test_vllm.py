from __future__ import annotations

import json
import threading
import time

import requests

from pllm.vllm import VLLMClient, is_standalone_vllm_command


def test_probe_and_level_one_round_trip(mock_vllm_url: str) -> None:
    client = VLLMClient(timeout=1.0)
    service = client.probe(mock_vllm_url)

    assert service.healthy
    assert service.controllable
    assert service.model == "mock-nemotron"

    client.sleep(service, level=1)
    client.wake(service, level=1)
    calls = requests.get(f"{mock_vllm_url}/calls", timeout=1).json()["calls"]

    assert calls[0] == {"method": "sleep", "level": 0, "mode": "keep"}
    assert calls[1] == {"method": "sleep", "level": 1, "mode": "keep"}
    assert calls[2] == {"method": "wake_up", "tags": []}


def test_level_two_wake_reloads_weights(mock_vllm_url: str) -> None:
    client = VLLMClient(timeout=1.0)
    service = client.probe(mock_vllm_url)

    client.sleep(service, level=2)
    client.wake(service, level=2)
    calls = requests.get(f"{mock_vllm_url}/calls", timeout=1).json()["calls"]

    assert calls[-3] == {"method": "wake_up", "tags": ["weights"]}
    assert calls[-2] == {
        "method": "collective_rpc",
        "payload": {"method": "reload_weights"},
    }
    assert calls[-1] == {"method": "wake_up", "tags": ["kv_cache"]}


def test_deep_sleep_after_explicit_quiesce_does_not_repeat_level_zero(
    mock_vllm_url: str,
) -> None:
    client = VLLMClient(timeout=1.0)
    service = client.probe(mock_vllm_url)

    client.sleep(service, level=0, mode="keep")
    client.sleep_from_quiesced(service, level=2, mode="keep")
    calls = requests.get(f"{mock_vllm_url}/calls", timeout=1).json()["calls"]

    assert calls == [
        {"method": "sleep", "level": 0, "mode": "keep"},
        {"method": "sleep", "level": 2, "mode": "keep"},
    ]


def test_level_zero_keep_freezes_and_resumes_same_stream(mock_vllm_url: str) -> None:
    client = VLLMClient(timeout=1.0)
    service = client.probe(mock_vllm_url)
    chunks: list[str] = []
    received = threading.Event()

    def consume() -> None:
        with requests.post(
            f"{mock_vllm_url}/v1/chat/completions",
            json={
                "model": "mock",
                "stream": True,
                "messages": [{"role": "user", "content": "one two three four"}],
            },
            stream=True,
            timeout=(1, 5),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                chunks.append(event["choices"][0]["delta"]["content"])
                received.set()

    thread = threading.Thread(target=consume)
    thread.start()
    assert received.wait(timeout=2)

    client.sleep(service, level=0, mode="keep")
    frozen_count = len(chunks)
    time.sleep(0.12)
    assert len(chunks) == frozen_count

    client.wake(service, level=0)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert "".join(chunks) == "Mock response: one two three four "


def test_training_commands_are_not_standalone_vllm() -> None:
    excluded = ("train", "grpo", "torchrun")

    assert is_standalone_vllm_command("vllm serve /models/qwen", excluded)
    assert not is_standalone_vllm_command(
        "python train_grpo.py --use-vllm vllm serve", excluded
    )
    assert not is_standalone_vllm_command("python train.py --vllm-util 0.2", excluded)
