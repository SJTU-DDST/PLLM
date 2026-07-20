from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .expert_catalog import read_safetensors_layout


SUPPORTED_VLLM = "0.25.1"
ROUTER_PROBE_PROFILE = "router_probe"
FULL_MTP_PROFILE = "full_mtp"
TARGET_GATE_PATTERN = re.compile(
    r"^backbone\.layers\.(?P<layer>\d+)\.mixer\.gate\."
    r"(?:weight|e_score_correction_bias)$"
)


@dataclass(slots=True, frozen=True)
class MTPTensor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    file_offset: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(slots=True, frozen=True)
class RouteMTPWeightPlan:
    profile: str
    tensors: tuple[MTPTensor, ...]
    total_mtp_bytes: int
    shared_components: tuple[str, ...]
    purpose: str

    @property
    def selected_bytes(self) -> int:
        return sum(item.size_bytes for item in self.tensors)

    @property
    def omitted_bytes(self) -> int:
        return self.total_mtp_bytes - self.selected_bytes

    @property
    def shards(self) -> tuple[str, ...]:
        return tuple(sorted({item.shard for item in self.tensors}))

    def summary(self, include_tensors: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "profile": self.profile,
            "purpose": self.purpose,
            "tensor_count": len(self.tensors),
            "selected_bytes": self.selected_bytes,
            "selected_mib": round(self.selected_bytes / 1024**2, 3),
            "total_mtp_bytes": self.total_mtp_bytes,
            "total_mtp_mib": round(self.total_mtp_bytes / 1024**2, 3),
            "omitted_bytes": self.omitted_bytes,
            "omitted_mib": round(self.omitted_bytes / 1024**2, 3),
            "shards": list(self.shards),
            "shared_components": list(self.shared_components),
            "loads_routed_expert_weights": any(
                ".mixer.experts." in item.name for item in self.tensors
            ),
            "produces_target_layer_routes": False,
            "requires_forecast_heads_or_calibration": True,
        }
        if include_tensors:
            payload["tensors"] = [item.to_dict() for item in self.tensors]
        return payload


@dataclass(slots=True, frozen=True)
class RouteMTPCompatibility:
    compatible: bool
    vllm_version: str
    implementation_path: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteMTPCheckpoint:
    model_path: Path
    architecture: str
    hidden_size: int
    expert_count: int
    active_experts: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    norm_epsilon: float
    expert_groups: int
    topk_groups: int
    prediction_steps: int
    pattern: str
    fingerprint: str
    tensors: tuple[MTPTensor, ...]
    target_gate_tensors: tuple[MTPTensor, ...]

    @classmethod
    def from_model(cls, model_path: str | Path) -> "RouteMTPCheckpoint":
        root = Path(model_path).expanduser().resolve()
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        config_bytes = config_path.read_bytes()
        index_bytes = index_path.read_bytes()
        config = json.loads(config_bytes)
        index = json.loads(index_bytes)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("safetensors index does not contain weight_map")

        prediction_steps = int(config.get("num_nextn_predict_layers", 0))
        pattern = str(config.get("mtp_hybrid_override_pattern", ""))
        if prediction_steps <= 0:
            raise ValueError("checkpoint does not enable multi-token prediction")
        if not pattern or any(char not in {"*", "E"} for char in pattern):
            raise ValueError(f"unsupported Nemotron MTP layer pattern: {pattern!r}")

        mtp_map = {
            str(name): str(shard)
            for name, shard in weight_map.items()
            if str(name).startswith("mtp.")
        }
        if not mtp_map:
            raise ValueError("checkpoint index does not contain mtp.* tensors")
        target_gate_map = {
            str(name): str(shard)
            for name, shard in weight_map.items()
            if TARGET_GATE_PATTERN.match(str(name))
        }

        layouts: dict[str, tuple[dict[str, Any], int]] = {}
        tensors: list[MTPTensor] = []
        target_gate_tensors: list[MTPTensor] = []
        all_selected = {**mtp_map, **target_gate_map}
        for name, shard_name in sorted(all_selected.items()):
            shard_path = root / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if shard_name not in layouts:
                layouts[shard_name] = read_safetensors_layout(shard_path)
            header, data_start = layouts[shard_name]
            metadata = header.get(name)
            if not isinstance(metadata, dict):
                raise ValueError(f"{name} is missing from {shard_name}")
            offsets = metadata.get("data_offsets")
            shape = metadata.get("shape")
            dtype = metadata.get("dtype")
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(value, int) for value in offsets)
                and isinstance(shape, list)
                and all(isinstance(value, int) and value >= 0 for value in shape)
                and isinstance(dtype, str)
            ):
                raise ValueError(f"invalid safetensors metadata for {name}")
            size_bytes = offsets[1] - offsets[0]
            if size_bytes < 0:
                raise ValueError(f"negative tensor size for {name}")
            tensor = MTPTensor(
                name=name,
                shard=shard_name,
                dtype=dtype,
                shape=tuple(shape),
                size_bytes=size_bytes,
                file_offset=data_start + offsets[0],
            )
            if name.startswith("mtp."):
                tensors.append(tensor)
            else:
                target_gate_tensors.append(tensor)

        architectures = config.get("architectures") or []
        result = cls(
            model_path=root,
            architecture=str(architectures[0]) if architectures else "",
            hidden_size=int(config.get("hidden_size", 0)),
            expert_count=int(config.get("n_routed_experts", 0)),
            active_experts=int(config.get("num_experts_per_tok", 0)),
            num_attention_heads=int(config.get("num_attention_heads", 0)),
            num_key_value_heads=int(config.get("num_key_value_heads", 0)),
            head_dim=int(config.get("head_dim", 0)),
            norm_epsilon=float(
                config.get("layer_norm_epsilon", config.get("norm_eps", 1e-5))
            ),
            expert_groups=int(config.get("n_group", 1)),
            topk_groups=int(config.get("topk_group", 1)),
            prediction_steps=prediction_steps,
            pattern=pattern,
            fingerprint=hashlib.sha256(config_bytes + b"\0" + index_bytes).hexdigest(),
            tensors=tuple(tensors),
            target_gate_tensors=tuple(target_gate_tensors),
        )
        result._validate_router_probe()
        return result

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.tensors)

    @property
    def routed_expert_bytes(self) -> int:
        return sum(
            item.size_bytes
            for item in self.tensors
            if ".mixer.experts." in item.name
        )

    @property
    def non_expert_bytes(self) -> int:
        return self.total_bytes - self.routed_expert_bytes

    def weight_plan(self, profile: str = ROUTER_PROBE_PROFILE) -> RouteMTPWeightPlan:
        normalized = profile.strip().lower()
        if normalized == FULL_MTP_PROFILE:
            selected = self.tensors
            purpose = "official_vllm_mtp_speculative_proposer"
        elif normalized == ROUTER_PROBE_PROFILE:
            gate_layer = self.pattern.index("E")
            prefixes = tuple(
                f"mtp.layers.{layer}." for layer in range(gate_layer)
            )
            gate_prefix = f"mtp.layers.{gate_layer}."
            selected = tuple(
                item
                for item in self.tensors
                if item.name.startswith(prefixes)
                or item.name == f"{gate_prefix}norm.weight"
                or item.name.startswith(f"{gate_prefix}mixer.gate.")
            )
            purpose = "mtp_future_state_to_mtp_router_probe_without_expert_execution"
        else:
            raise ValueError(
                f"unknown MTP profile {profile!r}; expected "
                f"{ROUTER_PROBE_PROFILE!r} or {FULL_MTP_PROFILE!r}"
            )
        return RouteMTPWeightPlan(
            profile=normalized,
            tensors=selected,
            total_mtp_bytes=self.total_bytes,
            shared_components=("target_hidden_states", "target_token_embedding"),
            purpose=purpose,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_path": str(self.model_path),
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "expert_count": self.expert_count,
            "active_experts": self.active_experts,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "prediction_steps": self.prediction_steps,
            "pattern": self.pattern,
            "fingerprint": self.fingerprint,
            "tensor_count": len(self.tensors),
            "total_bytes": self.total_bytes,
            "total_mib": round(self.total_bytes / 1024**2, 3),
            "routed_expert_bytes": self.routed_expert_bytes,
            "routed_expert_mib": round(self.routed_expert_bytes / 1024**2, 3),
            "non_expert_bytes": self.non_expert_bytes,
            "non_expert_mib": round(self.non_expert_bytes / 1024**2, 3),
            "router_probe": self.weight_plan().summary(),
            "target_route_heads": self.target_route_head_summary(),
            "evidence": "config_index_and_safetensors_headers_no_tensor_load",
            "gpu_allocated": False,
        }

    @property
    def target_route_layers(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    int(match.group("layer"))
                    for item in self.target_gate_tensors
                    if (match := TARGET_GATE_PATTERN.match(item.name)) is not None
                }
            )
        )

    def target_route_head_summary(self) -> dict[str, Any]:
        total = sum(item.size_bytes for item in self.target_gate_tensors)
        return {
            "tensor_count": len(self.target_gate_tensors),
            "layers": list(self.target_route_layers),
            "layer_count": len(self.target_route_layers),
            "bytes": total,
            "mib": round(total / 1024**2, 3),
            "initialization": "original_layer_gates_on_mtp_future_hidden_uncalibrated",
        }

    def _validate_router_probe(self) -> None:
        if self.pattern != "*E":
            raise ValueError(
                f"PLLM RouteMTP probe currently requires the audited '*E' pattern, "
                f"found {self.pattern!r}"
            )
        if self.prediction_steps != 1:
            raise ValueError(
                "PLLM RouteMTP currently follows vLLM's one-step Nemotron MTP support"
            )
        gate_layer = self.pattern.index("E")
        names = {item.name: item for item in self.tensors}
        required = {
            "mtp.layers.0.enorm.weight",
            "mtp.layers.0.hnorm.weight",
            "mtp.layers.0.eh_proj.weight",
            f"mtp.layers.{gate_layer}.norm.weight",
            f"mtp.layers.{gate_layer}.mixer.gate.weight",
            f"mtp.layers.{gate_layer}.mixer.gate.e_score_correction_bias",
        }
        missing = sorted(required - names.keys())
        if missing:
            raise ValueError(f"MTP router probe is missing tensors: {missing}")
        expected_shapes = {
            "mtp.layers.0.enorm.weight": (self.hidden_size,),
            "mtp.layers.0.hnorm.weight": (self.hidden_size,),
            "mtp.layers.0.eh_proj.weight": (self.hidden_size, self.hidden_size * 2),
            f"mtp.layers.{gate_layer}.norm.weight": (self.hidden_size,),
            f"mtp.layers.{gate_layer}.mixer.gate.weight": (
                self.expert_count,
                self.hidden_size,
            ),
            f"mtp.layers.{gate_layer}.mixer.gate.e_score_correction_bias": (
                self.expert_count,
            ),
        }
        for name, expected in expected_shapes.items():
            if names[name].shape != expected:
                raise ValueError(
                    f"unexpected shape for {name}: {names[name].shape}, expected {expected}"
                )
        if (
            self.num_attention_heads <= 0
            or self.num_key_value_heads <= 0
            or self.head_dim <= 0
            or self.num_attention_heads * self.head_dim != self.hidden_size
            or self.num_attention_heads % self.num_key_value_heads != 0
        ):
            raise ValueError("invalid MTP attention dimensions in model config")
        if (
            self.expert_groups <= 0
            or self.topk_groups <= 0
            or self.expert_count % self.expert_groups != 0
            or self.topk_groups > self.expert_groups
        ):
            raise ValueError("invalid MTP expert grouping in model config")
        expected_target_layers = len(self.target_route_layers)
        if not expected_target_layers or len(self.target_gate_tensors) != 2 * expected_target_layers:
            raise ValueError("checkpoint target route gate set is incomplete")
        for item in self.target_gate_tensors:
            expected = (
                (self.expert_count, self.hidden_size)
                if item.name.endswith(".gate.weight")
                else (self.expert_count,)
            )
            if item.shape != expected:
                raise ValueError(
                    f"unexpected shape for {item.name}: {item.shape}, expected {expected}"
                )


class RouteMTPTensorLoader:
    """Explicit, accelerator-deny-by-default loader for a validated MTP plan."""

    def __init__(self, checkpoint: RouteMTPCheckpoint) -> None:
        self.checkpoint = checkpoint

    def iter_tensors(
        self,
        profile: str = ROUTER_PROBE_PROFILE,
        *,
        device: str = "cpu",
        allow_accelerator: bool = False,
    ) -> Iterator[tuple[str, Any]]:
        normalized_device = str(device).strip().lower()
        if normalized_device != "cpu" and not allow_accelerator:
            raise ValueError(
                "accelerator tensor loading is disabled; pass allow_accelerator=True "
                "only during an explicit GPU experiment"
            )
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load MTP tensors") from exc

        plan = self.checkpoint.weight_plan(profile)
        yield from self._iter_selected(
            plan.tensors,
            normalized_device,
        )

    def iter_target_gate_tensors(
        self,
        *,
        device: str = "cpu",
        allow_accelerator: bool = False,
    ) -> Iterator[tuple[str, Any]]:
        normalized_device = str(device).strip().lower()
        if normalized_device != "cpu" and not allow_accelerator:
            raise ValueError(
                "accelerator tensor loading is disabled; pass allow_accelerator=True "
                "only during an explicit GPU experiment"
            )
        try:
            import safetensors  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load route heads") from exc
        yield from self._iter_selected(
            self.checkpoint.target_gate_tensors,
            normalized_device,
        )

    def _iter_selected(
        self,
        selected: tuple[MTPTensor, ...],
        device: str,
    ) -> Iterator[tuple[str, Any]]:
        from safetensors import safe_open

        by_shard: dict[str, list[MTPTensor]] = {}
        for tensor in selected:
            by_shard.setdefault(tensor.shard, []).append(tensor)
        for shard_name in sorted(by_shard):
            path = self.checkpoint.model_path / shard_name
            with safe_open(path, framework="pt", device=device) as handle:
                for tensor in by_shard[shard_name]:
                    yield tensor.name, handle.get_tensor(tensor.name)


def inspect_vllm_route_mtp_support(
    expected_version: str = SUPPORTED_VLLM,
) -> RouteMTPCompatibility:
    try:
        version = importlib.metadata.version("vllm")
        distribution = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        return RouteMTPCompatibility(False, "", "", "vLLM is not installed")
    implementation = Path(
        distribution.locate_file(
            "vllm/model_executor/models/nemotron_h_mtp.py"
        )
    )
    if version != expected_version:
        return RouteMTPCompatibility(
            False,
            version,
            str(implementation),
            f"PLLM requires the audited vLLM {expected_version} implementation",
        )
    if not implementation.is_file():
        return RouteMTPCompatibility(
            False,
            version,
            str(implementation),
            "NemotronH MTP implementation is absent",
        )
    source = implementation.read_text(encoding="utf-8")
    required_symbols = (
        "class NemotronHMultiTokenPredictor",
        "class NemotronHMTP",
        "mtp_hybrid_override_pattern",
    )
    missing = [symbol for symbol in required_symbols if symbol not in source]
    if missing:
        return RouteMTPCompatibility(
            False,
            version,
            str(implementation),
            f"audited MTP symbols are missing: {missing}",
        )
    return RouteMTPCompatibility(
        True,
        version,
        str(implementation),
        "official NemotronH speculative proposer is available",
    )
