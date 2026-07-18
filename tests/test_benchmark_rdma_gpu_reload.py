from __future__ import annotations

from pathlib import Path

from scripts.benchmark_rdma_gpu_reload import load_index


def test_load_index_resolves_profile_paths(tmp_path: Path) -> None:
    index = tmp_path / "profile.tsv"
    index.write_text("# slot\tkey\tsize\n0\tlayer-001/expert-0001.pllmex\t42\n")

    assert load_index(index, tmp_path) == [
        (tmp_path / "layer-001/expert-0001.pllmex", 42)
    ]
