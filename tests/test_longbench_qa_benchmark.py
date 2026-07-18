from __future__ import annotations

import pytest

from scripts.benchmark_longbench_qa import (
    best_qa_f1,
    directory_bytes,
    normalize_answer,
    parse_dataset_spec,
    percentile,
    qa_f1,
)


def test_normalize_answer_matches_english_qa_convention() -> None:
    assert normalize_answer("The South-West Ultras, fan club!") == (
        "southwest ultras fan club"
    )


def test_qa_f1_uses_token_overlap() -> None:
    assert qa_f1("South West Ultras", "The South West Ultras fan club") == pytest.approx(
        0.75
    )


def test_best_qa_f1_uses_best_reference() -> None:
    assert best_qa_f1("Fruits", ["Simple fruit", "Fruits", "Fruity"]) == 1.0


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_parse_dataset_spec() -> None:
    spec = parse_dataset_spec("mqa=test_data/mqa.jsonl=64")
    assert spec.name == "mqa"
    assert spec.path.as_posix() == "test_data/mqa.jsonl"
    assert spec.max_tokens == 64


def test_directory_bytes_counts_nested_files(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"123")
    (tmp_path / "nested" / "b.bin").write_bytes(b"4567")
    assert directory_bytes(tmp_path) == (7, 2)
