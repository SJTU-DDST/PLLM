from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from pllm.ssd_resume_pack import (
    ResumePackObject,
    build_resume_pack,
    load_resume_pack_manifest,
)


def test_resume_pack_is_contiguous_and_round_trips(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first-payload")
    second.write_bytes(b"second")
    destination = tmp_path / "resume.pack"

    manifest = build_resume_pack(
        [ResumePackObject(0, 3, first), ResumePackObject(2, 7, second)],
        destination,
        copy_chunk_bytes=4,
    )
    loaded = load_resume_pack_manifest(destination.with_suffix(".pack.json"))

    assert destination.read_bytes() == b"first-payloadsecond"
    assert loaded == manifest
    assert [(row.offset, row.size_bytes) for row in loaded.entries] == [
        (0, len(b"first-payload")),
        (len(b"first-payload"), len(b"second")),
    ]


def test_resume_pack_rejects_duplicate_keys(tmp_path: Path) -> None:
    source = tmp_path / "expert.bin"
    source.write_bytes(b"payload")

    with pytest.raises(ValueError, match="duplicate"):
        build_resume_pack(
            [ResumePackObject(0, 1, source), ResumePackObject(0, 1, source)],
            tmp_path / "resume.pack",
        )


def test_resume_pack_falls_back_across_filesystems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "expert.bin"
    source.write_bytes(b"cross-filesystem-payload")

    def reject_cross_filesystem(*_args: object) -> int:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(
        os, "copy_file_range", reject_cross_filesystem, raising=False
    )
    destination = tmp_path / "resume.pack"
    build_resume_pack(
        [ResumePackObject(0, 1, source)], destination, copy_chunk_bytes=4
    )

    assert destination.read_bytes() == source.read_bytes()
