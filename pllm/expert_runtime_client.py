from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .vllm_eer_runtime import request_runtime


class ExpertRuntimeClient:
    def __init__(self, socket_path: str, timeout_seconds: float = 30.0) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self.timeout_seconds = timeout_seconds
        self._cached_status: dict[str, Any] = {}
        self._cached_at = 0.0
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._cached_status and time.monotonic() - self._cached_at < 1.0:
                return dict(self._cached_status)
        try:
            result = self.request({"command": "status"}, timeout=1.0)
            result["online"] = True
            status = result
        except (OSError, RuntimeError, ValueError) as exc:
            status = {
                "online": False,
                "data_plane_ready": False,
                "backend": "vllm_eer_runtime_unavailable",
                "socket": str(self.socket_path),
                "error": str(exc),
            }
        with self._lock:
            self._cached_status = status
            self._cached_at = time.monotonic()
        return dict(status)

    def request(
        self, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        if not self.socket_path.exists():
            raise FileNotFoundError(f"EER runtime socket not found: {self.socket_path}")
        result = request_runtime(
            self.socket_path,
            payload,
            timeout=self.timeout_seconds if timeout is None else timeout,
        )
        if payload.get("command") != "status":
            with self._lock:
                self._cached_status = {}
                self._cached_at = 0.0
        return result

    def resize(
        self, slots_per_layer: int, retain_policy: str = "lru"
    ) -> dict[str, Any]:
        return self.request(
            {
                "command": "resize",
                "slots_per_layer": slots_per_layer,
                "retain_policy": retain_policy,
                "quiesced": True,
            },
            timeout=max(self.timeout_seconds, 600.0),
        )

    def set_phase(self, phase: str, reset_decode: bool = False) -> dict[str, Any]:
        return self.request(
            {
                "command": "phase",
                "phase": phase,
                "reset_decode": reset_decode,
            }
        )

    def prefetch(self, layer: int, experts: list[int]) -> dict[str, Any]:
        return self.request(
            {
                "command": "prefetch",
                "layer": layer,
                "experts": experts,
                "quiesced": True,
            }
        )

    def evict(self, layer: int, experts: list[int]) -> dict[str, Any]:
        return self.request(
            {
                "command": "evict",
                "layer": layer,
                "experts": experts,
                "quiesced": True,
            }
        )

    def evict_all(self) -> dict[str, Any]:
        return self.request({"command": "evict_all", "quiesced": True})

    def suspend(self) -> dict[str, Any]:
        return self.request({"command": "suspend", "quiesced": True})

    def resume(self) -> dict[str, Any]:
        return self.request({"command": "resume"})
