from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from pllm.expert_catalog import ExpertCatalog, read_safetensors_header
from pllm.expert_trace import ExpertRouteRecord, read_trace, synthetic_trace, write_trace


def _write_fake_model(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TinyMoE"],
                "num_hidden_layers": 2,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
            }
        ),
        encoding="utf-8",
    )
    header = {
        "backbone.embed.weight": {
            "dtype": "F16",
            "shape": [4, 4],
            "data_offsets": [0, 32],
        }
    }
    offset = 32
    for layer in (0, 1):
        for expert in range(4):
            for suffix in ("up_proj.weight", "down_proj.weight", "weight_scale"):
                size = 16 if suffix.endswith("weight") else 4
                header[
                    f"backbone.layers.{layer}.mixer.experts.{expert}.{suffix}"
                ] = {
                    "dtype": "F16",
                    "shape": [size // 2],
                    "data_offsets": [offset, offset + size],
                }
                offset += size
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(payload)) % 8
    payload += b" " * padding
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\0" * offset)
    return root


def test_catalog_parses_expert_objects_without_tensor_load(tmp_path: Path) -> None:
    model = _write_fake_model(tmp_path / "model")
    catalog = ExpertCatalog.from_model(model)

    assert catalog.architecture == "TinyMoE"
    assert catalog.moe_layers == [0, 1]
    assert len(catalog.experts) == 8
    assert catalog.routed_expert_bytes == 8 * 36
    assert catalog.non_routed_bytes == 32
    assert catalog.active_expert_bytes_per_token == catalog.routed_expert_bytes / 2
    assert catalog.project_slots(2)["resident_weight_bytes"] == 32 + 8 * 36 // 2
    assert read_safetensors_header(model / "model-00001-of-00001.safetensors")


def test_catalog_rejects_out_of_range_slots(tmp_path: Path) -> None:
    catalog = ExpertCatalog.from_model(_write_fake_model(tmp_path / "model"))
    with pytest.raises(ValueError):
        catalog.project_slots(5)


def test_trace_schema_round_trip_and_validation(tmp_path: Path) -> None:
    catalog = ExpertCatalog.from_model(_write_fake_model(tmp_path / "model"))
    records = list(synthetic_trace(catalog, requests=1, tokens_per_request=3))
    output = tmp_path / "trace.jsonl"

    count = write_trace(output, records, catalog)
    loaded = list(read_trace(output, catalog))

    assert count == 6
    assert [row.to_dict() for row in loaded] == [row.to_dict() for row in records]
    assert all(row.source == "synthetic_no_gpu" for row in loaded)

    invalid = ExpertRouteRecord(
        request_id="bad",
        workload="code",
        phase="decode",
        token_index=0,
        layer=0,
        actual_experts=[0],
    )
    with pytest.raises(ValueError):
        invalid.validate(catalog)
