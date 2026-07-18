from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pllm.expert_store import ExpertPackageCodec
from pllm.vllm_eer_runtime import request_runtime


EXPERT_PATH = re.compile(r"layer-(?P<layer>\d+)/expert-(?P<expert>\d+)\.pllmex")


def build_first_profile(root: Path, slots_per_layer: int) -> list[Path]:
    selected: list[Path] = []
    for layer_dir in sorted(root.glob("layer-*")):
        rows = sorted(layer_dir.glob("expert-*.pllmex"))[:slots_per_layer]
        if len(rows) != slots_per_layer:
            raise ValueError(f"{layer_dir.name} has only {len(rows)} experts")
        selected.extend(rows)
    if not selected:
        raise ValueError("no runtime expert objects found")
    return selected


def build_runtime_profile(root: Path, socket_path: Path) -> list[Path]:
    status = request_runtime(socket_path, {"command": "status"}, timeout=30)
    selected: list[Path] = []
    for layer in status.get("data_plane", {}).get("layers", []):
        layer_id = int(layer["layer"])
        snapshot = request_runtime(
            socket_path,
            {
                "command": "prefetch",
                "layer": layer_id,
                "experts": [],
                "quiesced": True,
            },
            timeout=30,
        )
        mapping = snapshot.get("status", {}).get("logical_to_slot", {})
        for expert in sorted(int(item) for item in mapping):
            selected.append(
                root / f"layer-{layer_id:03d}" / f"expert-{expert:04d}.pllmex"
            )
    if not selected:
        raise ValueError("the live EER runtime has no resident experts")
    return selected


def write_index(paths: list[Path], root: Path, output: Path) -> dict[str, Any]:
    rows = []
    total = 0
    for slot, path in enumerate(paths):
        relative = path.relative_to(root)
        size = path.stat().st_size
        rows.append(f"{slot}\t{relative.as_posix()}\t{size}")
        total += size
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# slot\tkey\tsize\n" + "\n".join(rows) + "\n")
    result = {"objects": len(paths), "bytes": total, "index": str(output)}
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def read_index(index_path: Path, root: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        _slot, key, size = line.split("\t")
        rows.append((root / key, int(size)))
    return rows


def split_index(index_path: Path, output_prefix: Path, shards: int) -> list[Path]:
    if shards <= 0:
        raise ValueError("shards must be positive")
    header = "# slot\tkey\tsize\n"
    rows = [
        line
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    outputs = []
    for shard in range(shards):
        output = output_prefix.parent / f"{output_prefix.name}-{shard:02d}.tsv"
        selected = rows[shard::shards]
        output.write_text(header + "\n".join(selected) + "\n", encoding="utf-8")
        outputs.append(output)
    return outputs


def validate_profile(index_path: Path, root: Path, fingerprint: str) -> dict[str, Any]:
    total = 0
    for path, expected_size in read_index(index_path, root):
        if path.stat().st_size != expected_size:
            raise ValueError(f"size mismatch: {path}")
        payload = ExpertPackageCodec.read(path)
        match = EXPERT_PATH.search(path.as_posix())
        if match is None:
            raise ValueError(f"invalid expert path: {path}")
        if (payload.layer, payload.expert) != (
            int(match.group("layer")),
            int(match.group("expert")),
        ):
            raise ValueError(f"expert identity mismatch: {path}")
        if fingerprint and payload.model_fingerprint != fingerprint:
            raise ValueError(f"model fingerprint mismatch: {path}")
        total += expected_size
    return {"validated": True, "objects": len(read_index(index_path, root)), "bytes": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate a PLLM RDMA memory-pool profile")
    parser.add_argument("command", choices=("build", "split", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--selection", choices=("first", "runtime"), default="first")
    parser.add_argument("--slots-per-layer", type=int, default=128)
    parser.add_argument("--socket", type=Path, default=Path("/tmp/pllm-2321/pllm-eer.sock"))
    parser.add_argument("--model-fingerprint", default="")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        paths = (
            build_first_profile(args.root, args.slots_per_layer)
            if args.selection == "first"
            else build_runtime_profile(args.root, args.socket)
        )
        result = write_index(paths, args.root, args.index)
    elif args.command == "split":
        prefix = args.output_prefix or args.index.with_suffix("")
        outputs = split_index(args.index, prefix, args.shards)
        result = {"shards": len(outputs), "outputs": [str(item) for item in outputs]}
    else:
        result = validate_profile(args.index, args.root, args.model_fingerprint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
