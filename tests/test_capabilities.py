from __future__ import annotations

from pathlib import Path

from pllm.capabilities import CapabilityProbe
from pllm.config import PLLMConfig
from pllm.hibercache import HiberCacheManager


def test_unknown_gpudirect_capability_uses_host_staging(
    tmp_path: Path, monkeypatch
) -> None:
    config = PLLMConfig(
        model_path=str(tmp_path / "model"),
        hibercache_dir=str(tmp_path / "cache"),
    )
    probe = CapabilityProbe(config, HiberCacheManager(config))
    monkeypatch.setattr(probe, "_gpu_name", lambda: "Unknown NVIDIA GPU")
    monkeypatch.setattr(probe, "_cuda_attribute", lambda _attribute: None)
    monkeypatch.setattr(probe, "_rdma_devices", lambda: [])

    result = probe.collect(refresh=True)

    assert result["cuda"]["gpudirect_rdma"] is None
    assert result["cuda"]["rdma_path"] == "host_staging"
