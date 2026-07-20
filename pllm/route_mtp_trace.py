from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .route_mtp_runtime import RouteMTPPrediction


class RouteMTPTraceWriter:
    """Request-scoped, off-critical-path RouteMTP calibration traces."""

    def __init__(
        self,
        root: str | Path,
        *,
        layers: Sequence[int],
        hidden_size: int,
        active_experts: int,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.layers = tuple(int(layer) for layer in layers)
        self.hidden_size = int(hidden_size)
        self.active_experts = int(active_experts)
        self._records: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._sequence = 0
        self.written_requests = 0
        self.written_samples = 0
        self.written_bytes = 0

    def append(
        self,
        prediction: RouteMTPPrediction,
        actual_by_layer: Mapping[int, Sequence[int]],
    ) -> None:
        if prediction.route_hidden is None:
            raise ValueError("RouteMTP trace capture requires route_hidden")
        feature = np.asarray(prediction.route_hidden, dtype=np.float16)
        if feature.shape != (self.hidden_size,):
            raise ValueError(
                f"route_hidden must have shape ({self.hidden_size},), got {feature.shape}"
            )
        labels = np.asarray(
            [actual_by_layer[layer] for layer in self.layers], dtype=np.uint16
        )
        expected = (len(self.layers), self.active_experts)
        if labels.shape != expected:
            raise ValueError(f"target routes must have shape {expected}, got {labels.shape}")
        mtp = np.asarray(prediction.mtp_experts, dtype=np.uint16)
        if mtp.shape != (self.active_experts,):
            raise ValueError(
                "MTP route must contain exactly "
                f"{self.active_experts} experts, got {mtp.shape}"
            )
        self._records.setdefault(prediction.request_id, []).append(
            (
                int(prediction.token_id),
                np.array(feature, copy=True),
                np.array(labels, copy=True),
                np.array(mtp, copy=True),
            )
        )

    def flush(self) -> dict[str, Any]:
        paths = []
        samples = 0
        for request_id, records in list(self._records.items()):
            if not records:
                continue
            path = self._write_request(request_id, records)
            paths.append(str(path))
            samples += len(records)
            del self._records[request_id]
        return {
            "files": paths,
            "requests": len(paths),
            "samples": samples,
            **self.status(),
        }

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "root": str(self.root),
            "pending_requests": len(self._records),
            "pending_samples": sum(len(records) for records in self._records.values()),
            "written_requests": self.written_requests,
            "written_samples": self.written_samples,
            "written_bytes": self.written_bytes,
            "format": "request_scoped_npz_uncompressed",
        }

    def _write_request(
        self,
        request_id: str,
        records: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    ) -> Path:
        self._sequence += 1
        digest = hashlib.sha256(request_id.encode()).hexdigest()[:12]
        path = self.root / f"trace-{self._sequence:06d}-{digest}.npz"
        temporary = path.with_suffix(".npz.tmp")
        metadata = {
            "schema_version": 1,
            "request_id": request_id,
            "created_at_unix": time.time(),
            "layers": list(self.layers),
            "hidden_size": self.hidden_size,
            "active_experts": self.active_experts,
            "alignment": "feature_at_token_t_predicts_target_routes_at_token_t_plus_1",
            "feature": "mtp_pre_moe_route_hidden_fp16",
        }
        token_ids = np.asarray([record[0] for record in records], dtype=np.int32)
        features = np.stack([record[1] for record in records])
        actual_routes = np.stack([record[2] for record in records])
        mtp_routes = np.stack([record[3] for record in records])
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
                token_ids=token_ids,
                features=features,
                actual_routes=actual_routes,
                mtp_routes=mtp_routes,
            )
            handle.flush()
        temporary.replace(path)
        size = path.stat().st_size
        self.written_requests += 1
        self.written_samples += len(records)
        self.written_bytes += size
        return path
