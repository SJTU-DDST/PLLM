from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class RouteTraceRequest:
    path: Path
    request_id: str
    layers: tuple[int, ...]
    features: np.ndarray
    actual_routes: np.ndarray
    mtp_routes: np.ndarray

    @property
    def samples(self) -> int:
        return int(self.features.shape[0])


def load_route_trace(path: str | Path) -> RouteTraceRequest:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        features = np.array(payload["features"], dtype=np.float16, copy=True)
        actual_routes = np.array(
            payload["actual_routes"], dtype=np.uint16, copy=True
        )
        mtp_routes = np.array(payload["mtp_routes"], dtype=np.uint16, copy=True)
    layers = tuple(int(layer) for layer in metadata["layers"])
    hidden_size = int(metadata["hidden_size"])
    active_experts = int(metadata["active_experts"])
    samples = features.shape[0]
    if features.shape != (samples, hidden_size):
        raise ValueError(f"invalid feature shape in {source}: {features.shape}")
    expected_routes = (samples, len(layers), active_experts)
    if actual_routes.shape != expected_routes:
        raise ValueError(
            f"invalid target route shape in {source}: {actual_routes.shape}"
        )
    if mtp_routes.shape != (samples, active_experts):
        raise ValueError(f"invalid MTP route shape in {source}: {mtp_routes.shape}")
    return RouteTraceRequest(
        path=source,
        request_id=str(metadata["request_id"]),
        layers=layers,
        features=features,
        actual_routes=actual_routes,
        mtp_routes=mtp_routes,
    )


def load_route_traces(paths: Iterable[str | Path]) -> list[RouteTraceRequest]:
    requests = [load_route_trace(path) for path in paths]
    if not requests:
        raise ValueError("no RouteMTP traces were provided")
    reference = (
        requests[0].layers,
        requests[0].features.shape[1],
        requests[0].actual_routes.shape[2],
    )
    for request in requests[1:]:
        current = (
            request.layers,
            request.features.shape[1],
            request.actual_routes.shape[2],
        )
        if current != reference:
            raise ValueError("RouteMTP traces use incompatible model dimensions")
    return requests


class SharedLowRankRouteAdapter(nn.Module):
    """Shared MTP projection with independent target-layer route heads."""

    def __init__(
        self,
        hidden_size: int,
        layer_count: int,
        expert_count: int,
        rank: int,
    ) -> None:
        super().__init__()
        if min(hidden_size, layer_count, expert_count, rank) <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.layer_count = int(layer_count)
        self.expert_count = int(expert_count)
        self.rank = int(rank)
        self.projection = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.layer_weights = nn.Parameter(
            torch.empty(self.layer_count, self.expert_count, self.rank)
        )
        self.layer_bias = nn.Parameter(
            torch.zeros(self.layer_count, self.expert_count)
        )
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.layer_weights)

    def forward(self, route_hidden: torch.Tensor) -> torch.Tensor:
        if route_hidden.ndim != 2 or route_hidden.shape[1] != self.hidden_size:
            raise ValueError(
                f"route_hidden must have shape [batch, {self.hidden_size}]"
            )
        hidden = route_hidden.float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1e-6)
        latent = torch.nn.functional.gelu(self.projection(hidden))
        return torch.einsum("br,ler->ble", latent, self.layer_weights) + self.layer_bias

    def config(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "layer_count": self.layer_count,
            "expert_count": self.expert_count,
            "rank": self.rank,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model": self.config(),
                "state_dict": self.state_dict(),
                "metadata": metadata,
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str = "cpu") -> tuple["SharedLowRankRouteAdapter", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**payload["model"])
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("metadata", {}))
