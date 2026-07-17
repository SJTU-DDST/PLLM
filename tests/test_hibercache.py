from __future__ import annotations

import os
from pathlib import Path

from pllm.config import PLLMConfig
from pllm.hibercache import HiberCacheManager


def test_transfer_config_and_lru_quota(tmp_path: Path) -> None:
    config = PLLMConfig(
        hibercache_dir=str(tmp_path / "cache"),
        hibercache_staging_mb=512,
        hibercache_quota_gb=0.000001,
    )
    cache = HiberCacheManager(config)
    older = cache.root / "older.block"
    newer = cache.root / "newer.block"
    older.write_bytes(b"a" * 900)
    newer.write_bytes(b"b" * 900)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    spec = cache.vllm_transfer_config()
    status = cache.enforce_quota()

    assert spec["kv_connector"] == "OffloadingConnector"
    assert spec["kv_connector_extra_config"]["cpu_bytes_to_use"] == 512 * 1024**2
    assert not older.exists()
    assert newer.exists()
    assert status["used_gb"] <= config.hibercache_quota_gb
