from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


GIB = 1024**3
MIB = 1024**2
EXPERT_PATTERN = re.compile(
    r"^backbone\.layers\.(?P<layer>\d+)\.mixer\.experts\."
    r"(?P<expert>\d+)\."
)


@dataclass(slots=True)
class TensorSlice:
    name: str
    shard: str
    dtype: str
    shape: list[int]
    data_offset: int
    file_offset: int
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpertObject:
    layer: int
    expert: int
    size_bytes: int = 0
    tensors: list[str] = field(default_factory=list)
    shards: list[str] = field(default_factory=list)
    slices: list[TensorSlice] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpertCatalog:
    model_path: str
    architecture: str
    num_hidden_layers: int
    moe_layers: list[int]
    experts_per_layer: int
    active_experts_per_token: int
    total_tensor_bytes: int
    routed_expert_bytes: int
    non_routed_bytes: int
    experts: list[ExpertObject]

    @classmethod
    def from_model(cls, model_path: str | Path) -> "ExpertCatalog":
        root = Path(model_path).expanduser().resolve()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        objects: dict[tuple[int, int], ExpertObject] = {}
        total_tensor_bytes = 0
        routed_expert_bytes = 0

        for shard in sorted(root.glob("*.safetensors")):
            header, data_start = read_safetensors_layout(shard)
            for tensor_name, tensor in header.items():
                if tensor_name == "__metadata__" or not isinstance(tensor, dict):
                    continue
                offsets = tensor.get("data_offsets")
                if not (
                    isinstance(offsets, list)
                    and len(offsets) == 2
                    and all(isinstance(value, int) for value in offsets)
                ):
                    raise ValueError(f"invalid data_offsets for {tensor_name} in {shard}")
                size_bytes = offsets[1] - offsets[0]
                if size_bytes < 0:
                    raise ValueError(f"negative tensor size for {tensor_name} in {shard}")
                total_tensor_bytes += size_bytes
                match = EXPERT_PATTERN.match(tensor_name)
                if match is None:
                    continue
                layer = int(match.group("layer"))
                expert = int(match.group("expert"))
                obj = objects.setdefault(
                    (layer, expert), ExpertObject(layer=layer, expert=expert)
                )
                obj.size_bytes += size_bytes
                obj.tensors.append(tensor_name)
                if shard.name not in obj.shards:
                    obj.shards.append(shard.name)
                dtype = tensor.get("dtype")
                shape = tensor.get("shape")
                if not isinstance(dtype, str) or not (
                    isinstance(shape, list)
                    and all(isinstance(value, int) and value >= 0 for value in shape)
                ):
                    raise ValueError(f"invalid tensor metadata for {tensor_name}")
                obj.slices.append(
                    TensorSlice(
                        name=tensor_name,
                        shard=shard.name,
                        dtype=dtype,
                        shape=shape,
                        data_offset=offsets[0],
                        file_offset=data_start + offsets[0],
                        size_bytes=size_bytes,
                    )
                )
                routed_expert_bytes += size_bytes

        experts = sorted(objects.values(), key=lambda item: (item.layer, item.expert))
        moe_layers = sorted({item.layer for item in experts})
        expected_experts = int(config.get("n_routed_experts", 0))
        if expected_experts and any(
            sum(item.layer == layer for item in experts) != expected_experts
            for layer in moe_layers
        ):
            raise ValueError("checkpoint does not contain a complete expert set per MoE layer")
        architectures = config.get("architectures") or []
        return cls(
            model_path=str(root),
            architecture=str(architectures[0]) if architectures else "",
            num_hidden_layers=int(config.get("num_hidden_layers", 0)),
            moe_layers=moe_layers,
            experts_per_layer=expected_experts,
            active_experts_per_token=int(config.get("num_experts_per_tok", 0)),
            total_tensor_bytes=total_tensor_bytes,
            routed_expert_bytes=routed_expert_bytes,
            non_routed_bytes=total_tensor_bytes - routed_expert_bytes,
            experts=experts,
        )

    @property
    def average_expert_bytes(self) -> float:
        return self.routed_expert_bytes / len(self.experts) if self.experts else 0.0

    @property
    def active_expert_bytes_per_token(self) -> float:
        if not self.experts_per_layer:
            return 0.0
        return self.routed_expert_bytes * (
            self.active_experts_per_token / self.experts_per_layer
        )

    def project_slots(self, slots_per_layer: int) -> dict[str, Any]:
        if slots_per_layer < 0 or slots_per_layer > self.experts_per_layer:
            raise ValueError(
                f"slots_per_layer must be between 0 and {self.experts_per_layer}"
            )
        routed_bytes = int(
            round(self.routed_expert_bytes * slots_per_layer / self.experts_per_layer)
        ) if self.experts_per_layer else 0
        resident_bytes = self.non_routed_bytes + routed_bytes
        return {
            "slots_per_layer": slots_per_layer,
            "routed_bytes": routed_bytes,
            "resident_weight_bytes": resident_bytes,
            "projected_reclaim_bytes": self.total_tensor_bytes - resident_bytes,
            "resident_weight_gib": round(resident_bytes / GIB, 3),
            "projected_reclaim_gib": round(
                (self.total_tensor_bytes - resident_bytes) / GIB, 3
            ),
            "evidence": "checkpoint_header_projection",
        }

    def summary(self, include_experts: bool = False) -> dict[str, Any]:
        sizes = [item.size_bytes for item in self.experts]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "model_path": self.model_path,
            "architecture": self.architecture,
            "num_hidden_layers": self.num_hidden_layers,
            "moe_layers": self.moe_layers,
            "moe_layer_count": len(self.moe_layers),
            "experts_per_layer": self.experts_per_layer,
            "active_experts_per_token": self.active_experts_per_token,
            "expert_object_count": len(self.experts),
            "total_tensor_bytes": self.total_tensor_bytes,
            "routed_expert_bytes": self.routed_expert_bytes,
            "non_routed_bytes": self.non_routed_bytes,
            "average_expert_bytes": round(self.average_expert_bytes, 3),
            "min_expert_bytes": min(sizes, default=0),
            "max_expert_bytes": max(sizes, default=0),
            "active_expert_bytes_per_token": round(
                self.active_expert_bytes_per_token, 3
            ),
            "projections": [
                self.project_slots(slots)
                for slots in (32, 64, 128, 256, self.experts_per_layer)
                if slots <= self.experts_per_layer
            ],
            "evidence": {
                "source": "safetensors_headers_and_config",
                "gpu_allocated": False,
                "runtime_release_measured": False,
            },
        }
        if include_experts:
            payload["experts"] = [item.to_dict() for item in self.experts]
        return payload


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    return read_safetensors_layout(path)[0]


def read_safetensors_layout(path: str | Path) -> tuple[dict[str, Any], int]:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header in {file_path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > file_path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length in {file_path}")
        payload = handle.read(header_length)
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header in {file_path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object in {file_path}")
    return header, 8 + header_length
