from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from pllm.route_mtp import (
    FULL_MTP_PROFILE,
    ROUTER_PROBE_PROFILE,
    RouteMTPCheckpoint,
    RouteMTPTensorLoader,
)


def _write_shard(path: Path, tensors: dict[str, tuple[str, list[int], int]]) -> None:
    header = {}
    offset = 0
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


def _write_fake_mtp(root: Path) -> Path:
    root.mkdir()
    config = {
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": 4,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "layer_norm_epsilon": 1e-5,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*E",
        "hybrid_override_pattern": "E*E",
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    first = {
        "backbone.layers.0.mixer.gate.weight": ("BF16", [8, 4], 64),
        "backbone.layers.0.mixer.gate.e_score_correction_bias": ("F32", [8], 32),
        "mtp.layers.0.enorm.weight": ("BF16", [4], 8),
        "mtp.layers.0.hnorm.weight": ("BF16", [4], 8),
        "mtp.layers.0.eh_proj.weight": ("BF16", [4, 8], 64),
        "mtp.layers.0.mixer.q_proj.weight": ("BF16", [4, 4], 32),
        "mtp.layers.0.mixer.k_proj.weight": ("BF16", [2, 4], 16),
        "mtp.layers.0.mixer.v_proj.weight": ("BF16", [2, 4], 16),
        "mtp.layers.0.mixer.o_proj.weight": ("BF16", [4, 4], 32),
        "mtp.layers.0.norm.weight": ("BF16", [4], 8),
        "mtp.layers.1.norm.weight": ("BF16", [4], 8),
    }
    second = {
        "backbone.layers.2.mixer.gate.weight": ("BF16", [8, 4], 64),
        "backbone.layers.2.mixer.gate.e_score_correction_bias": ("F32", [8], 32),
        "mtp.layers.1.mixer.gate.weight": ("BF16", [8, 4], 64),
        "mtp.layers.1.mixer.gate.e_score_correction_bias": ("F32", [8], 32),
        "mtp.layers.1.mixer.experts.0.up_proj.weight": ("BF16", [4, 4], 32),
        "mtp.layers.1.mixer.experts.0.down_proj.weight": ("BF16", [4, 4], 32),
        "mtp.layers.1.final_layernorm.weight": ("BF16", [4], 8),
    }
    shard_a = "model-00001-of-00002.safetensors"
    shard_b = "model-00002-of-00002.safetensors"
    _write_shard(root / shard_a, first)
    _write_shard(root / shard_b, second)
    weight_map = {name: shard_a for name in first}
    weight_map.update({name: shard_b for name in second})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return root


def test_mtp_checkpoint_builds_router_probe_without_loading_experts(
    tmp_path: Path,
) -> None:
    checkpoint = RouteMTPCheckpoint.from_model(_write_fake_mtp(tmp_path / "model"))
    probe = checkpoint.weight_plan(ROUTER_PROBE_PROFILE)
    full = checkpoint.weight_plan(FULL_MTP_PROFILE)

    assert checkpoint.pattern == "*E"
    assert checkpoint.expert_count == 8
    assert checkpoint.target_route_layers == (0, 2)
    assert len(full.tensors) == 14
    assert len(probe.tensors) == 11
    assert probe.selected_bytes < full.selected_bytes
    assert not any(".mixer.experts." in item.name for item in probe.tensors)
    assert probe.summary()["produces_target_layer_routes"] is False
    assert checkpoint.summary()["gpu_allocated"] is False
    assert checkpoint.target_route_head_summary()["tensor_count"] == 4


def test_mtp_checkpoint_rejects_an_incompatible_gate_shape(tmp_path: Path) -> None:
    model = _write_fake_mtp(tmp_path / "model")
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = model / index["weight_map"]["mtp.layers.1.mixer.gate.weight"]
    header_length = struct.unpack("<Q", shard.read_bytes()[:8])[0]
    raw = shard.read_bytes()
    header = json.loads(raw[8 : 8 + header_length])
    header["mtp.layers.1.mixer.gate.weight"]["shape"] = [7, 4]
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    data = raw[8 + header_length :]
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)

    with pytest.raises(ValueError, match="unexpected shape"):
        RouteMTPCheckpoint.from_model(model)


def test_mtp_tensor_loader_denies_accelerator_by_default(tmp_path: Path) -> None:
    checkpoint = RouteMTPCheckpoint.from_model(_write_fake_mtp(tmp_path / "model"))
    loader = RouteMTPTensorLoader(checkpoint)

    with pytest.raises(ValueError, match="accelerator tensor loading is disabled"):
        next(loader.iter_tensors(device="cuda"))


def test_mtp_tensor_loader_reads_only_the_cpu_probe_profile(tmp_path: Path) -> None:
    checkpoint = RouteMTPCheckpoint.from_model(_write_fake_mtp(tmp_path / "model"))
    loaded = dict(RouteMTPTensorLoader(checkpoint).iter_tensors())

    assert len(loaded) == 11
    assert loaded["mtp.layers.0.eh_proj.weight"].device.type == "cpu"
    assert tuple(loaded["mtp.layers.0.eh_proj.weight"].shape) == (4, 8)
    assert not any(".mixer.experts." in name for name in loaded)
