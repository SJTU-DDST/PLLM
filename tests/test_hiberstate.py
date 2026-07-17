from __future__ import annotations

from pathlib import Path

import pytest

from pllm.hiberstate import HiberStateSnapshot, HiberStateStore, StateComponent


def test_transactional_live_state_round_trip(tmp_path: Path) -> None:
    store = HiberStateStore(tmp_path, chunk_bytes=4)
    snapshot = HiberStateSnapshot(
        request_id="request-1",
        epoch=7,
        model_fingerprint="f" * 64,
        committed_tokens=197,
        components=(
            StateComponent("attention-kv", b"abcdefgh", "fp8", (2, 4)),
            StateComponent("mamba-state", b"ijklmnop", "bf16", (4,)),
            StateComponent("sampler-ledger", b"qrst", metadata={"rng": 3}),
        ),
    )

    committed = store.commit(snapshot)
    restored = store.load("request-1", 7, "f" * 64)

    assert committed["committed_tokens"] == 197
    assert restored == snapshot


def test_corrupt_live_state_chunk_fails_closed(tmp_path: Path) -> None:
    store = HiberStateStore(tmp_path, chunk_bytes=4)
    snapshot = HiberStateSnapshot(
        request_id="request-2",
        epoch=1,
        model_fingerprint="f" * 64,
        committed_tokens=3,
        components=(StateComponent("attention-kv", b"abcdefgh"),),
    )
    store.commit(snapshot)
    chunk = next((tmp_path / "request-2").glob("epoch-*/*.chunk"))
    chunk.write_bytes(b"bad")

    with pytest.raises(ValueError, match="chunk"):
        store.load("request-2", 1)
