from __future__ import annotations

from pathlib import Path

from scripts.rdma_memory_profile import (
    build_first_profile,
    read_index,
    split_index,
    write_index,
)


def test_memory_profile_uses_fixed_slots_per_layer(tmp_path: Path) -> None:
    root = tmp_path / "experts"
    for layer in (1, 3):
        directory = root / f"layer-{layer:03d}"
        directory.mkdir(parents=True)
        for expert in range(4):
            (directory / f"expert-{expert:04d}.pllmex").write_bytes(
                bytes([layer, expert])
            )

    selected = build_first_profile(root, 2)
    index = tmp_path / "profile.tsv"
    result = write_index(selected, root, index)

    assert result["objects"] == 4
    assert [path.relative_to(root).as_posix() for path, _ in read_index(index, root)] == [
        "layer-001/expert-0000.pllmex",
        "layer-001/expert-0001.pllmex",
        "layer-003/expert-0000.pllmex",
        "layer-003/expert-0001.pllmex",
    ]

    shards = split_index(index, tmp_path / "profile-shard", 2)
    assert "\tlayer-001/expert-0000.pllmex\t" in shards[0].read_text()
    assert "\tlayer-001/expert-0001.pllmex\t" in shards[1].read_text()
