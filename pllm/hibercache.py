from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from .config import PLLMConfig
from .expert_store import RDMABridgeTransport
from .hiberstate import HiberStateSnapshot, HiberStateStore


class HiberCacheManager:
    def __init__(self, config: PLLMConfig) -> None:
        self.config = config
        self.root = Path(config.hibercache_dir).expanduser()
        self._last_scan_at = 0.0
        self._cached_status: dict[str, Any] = {}
        self.last_error = ""
        transport = None
        if config.hiberstate_rdma_enabled:
            transport = RDMABridgeTransport(
                config.hiberstate_rdma_peer,
                config.hiberstate_rdma_port,
                config.hiberstate_rdma_binary,
                token_file=config.hiberstate_rdma_token_file,
                allocator=config.hiberstate_rdma_allocator,
                device=config.hiberstate_rdma_device,
                ib_port=config.hiberstate_rdma_ib_port,
                gid_index=config.hiberstate_rdma_gid_index,
            )
        self.state_store = HiberStateStore(
            self.root / "live-state",
            chunk_bytes=int(config.hiberstate_chunk_mb * 1024**2),
            transport=transport,
        )
        if config.hibercache_enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.last_error = str(exc)

    @property
    def quota_bytes(self) -> int:
        return int(self.config.hibercache_quota_gb * 1024**3)

    def vllm_transfer_config(self) -> dict[str, Any]:
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": int(self.config.hibercache_staging_mb * 1024**2),
                "eviction_policy": "arc",
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(self.root),
                        "n_read_threads": 8,
                        "n_write_threads": 4,
                    }
                ],
            },
        }

    def status(self, refresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not refresh and self._cached_status and now - self._last_scan_at < 5.0:
            return dict(self._cached_status)
        if self.root.is_dir() and os.access(self.root, os.R_OK | os.W_OK | os.X_OK):
            self.last_error = ""
        files = 0
        used = 0
        newest = 0.0
        if self.root.exists():
            try:
                for path in self.root.rglob("*"):
                    if not path.is_file():
                        continue
                    stat = path.stat()
                    files += 1
                    used += stat.st_size
                    newest = max(newest, stat.st_mtime)
            except OSError as exc:
                self.last_error = str(exc)
        self._cached_status = {
            "enabled": self.config.hibercache_enabled and not self.last_error,
            "root": str(self.root),
            "staging_mb": self.config.hibercache_staging_mb,
            "quota_gb": self.config.hibercache_quota_gb,
            "used_gb": round(used / 1024**3, 3),
            "files": files,
            "newest_at": newest or None,
            "load_failure_policy": "recompute",
            "exact_cache_dtype": "fp8",
            "attention_state": "connector_tiered_by_cache_block",
            "recurrent_state": "connector_tiered_mamba_page",
            "active_state_island": {
                "weight_independent": True,
                "attention_kv": "OffloadingConnector AttentionSpec",
                "mamba_conv_ssm": "OffloadingConnector MambaSpec",
                "deep_sleep_exact_resume_validated": False,
            },
            "live_state_store": self.state_store.status(),
            "real_model_continuity_validated": False,
            "error": self.last_error,
        }
        self._last_scan_at = now
        return dict(self._cached_status)

    def enforce_quota(self) -> dict[str, Any]:
        if not self.root.exists() or self.quota_bytes <= 0:
            return self.status(refresh=True)
        candidates: list[tuple[float, int, Path, bool]] = []
        total = 0
        for path in self.root.rglob("*"):
            if not path.is_file() or path.name.endswith("config.json"):
                continue
            if self.state_store.root in path.parents:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            total += stat.st_size
            candidates.append((stat.st_mtime, stat.st_size, path, False))
        for atime, size, path in self.state_store.transaction_entries():
            total += size
            candidates.append((atime, size, path, True))
        for _mtime, size, path, is_directory in sorted(candidates):
            if total <= self.quota_bytes:
                break
            try:
                if is_directory:
                    shutil.rmtree(path)
                    try:
                        path.parent.rmdir()
                    except OSError:
                        pass
                else:
                    os.remove(path)
                total -= size
            except OSError:
                continue
        return self.status(refresh=True)

    def commit_live_state(
        self, snapshot: HiberStateSnapshot, replicate: bool | None = None
    ) -> dict[str, Any]:
        result = self.state_store.commit(
            snapshot,
            replicate=(
                self.config.hiberstate_rdma_enabled
                if replicate is None
                else replicate
            ),
        )
        self._cached_status = {}
        return result

    def load_live_state(
        self,
        request_id: str,
        epoch: int | None = None,
        expected_model_fingerprint: str = "",
    ) -> HiberStateSnapshot:
        return self.state_store.load(
            request_id, epoch, expected_model_fingerprint
        )
