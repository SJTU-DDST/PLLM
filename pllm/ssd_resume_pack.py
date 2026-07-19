from __future__ import annotations

import errno
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


PACK_SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class ResumePackObject:
    layer: int
    expert: int
    source: Path


@dataclass(slots=True, frozen=True)
class ResumePackEntry:
    layer: int
    expert: int
    offset: int
    size_bytes: int


@dataclass(slots=True, frozen=True)
class ResumePackManifest:
    schema_version: int
    pack_name: str
    size_bytes: int
    entries: tuple[ResumePackEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pack_name": self.pack_name,
            "size_bytes": self.size_bytes,
            "entries": [asdict(entry) for entry in self.entries],
        }


def build_resume_pack(
    objects: Sequence[ResumePackObject],
    pack_path: Path,
    manifest_path: Path | None = None,
    *,
    copy_chunk_bytes: int = 64 * 1024**2,
) -> ResumePackManifest:
    if not objects:
        raise ValueError("resume pack cannot be empty")
    if copy_chunk_bytes <= 0:
        raise ValueError("copy_chunk_bytes must be positive")
    keys = [(int(item.layer), int(item.expert)) for item in objects]
    if len(keys) != len(set(keys)):
        raise ValueError("resume pack contains duplicate expert keys")
    for item in objects:
        if not item.source.is_file():
            raise FileNotFoundError(item.source)

    pack_path = Path(pack_path)
    manifest_path = manifest_path or pack_path.with_suffix(pack_path.suffix + ".json")
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pack_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{pack_path.name}.", dir=pack_path.parent
    )
    temporary_pack = Path(temporary_name)
    entries: list[ResumePackEntry] = []
    offset = 0
    try:
        for item in objects:
            size = item.source.stat().st_size
            source_fd = os.open(item.source, os.O_RDONLY)
            try:
                _copy_exact(
                    source_fd,
                    pack_fd,
                    size,
                    copy_chunk_bytes,
                    item.source,
                )
            finally:
                os.close(source_fd)
            entries.append(
                ResumePackEntry(
                    layer=int(item.layer),
                    expert=int(item.expert),
                    offset=offset,
                    size_bytes=size,
                )
            )
            offset += size
        os.fsync(pack_fd)
        os.close(pack_fd)
        pack_fd = -1
        os.replace(temporary_pack, pack_path)
        _fsync_directory(pack_path.parent)

        manifest = ResumePackManifest(
            schema_version=PACK_SCHEMA_VERSION,
            pack_name=pack_path.name,
            size_bytes=offset,
            entries=tuple(entries),
        )
        _write_manifest(manifest_path, manifest)
        return manifest
    finally:
        if pack_fd >= 0:
            os.close(pack_fd)
        temporary_pack.unlink(missing_ok=True)


def load_resume_pack_manifest(path: Path) -> ResumePackManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != PACK_SCHEMA_VERSION:
        raise ValueError("unsupported resume pack schema")
    entries = tuple(
        ResumePackEntry(
            layer=int(item["layer"]),
            expert=int(item["expert"]),
            offset=int(item["offset"]),
            size_bytes=int(item["size_bytes"]),
        )
        for item in payload.get("entries", [])
    )
    manifest = ResumePackManifest(
        schema_version=PACK_SCHEMA_VERSION,
        pack_name=str(payload["pack_name"]),
        size_bytes=int(payload["size_bytes"]),
        entries=entries,
    )
    expected_offset = 0
    for entry in manifest.entries:
        if entry.offset != expected_offset or entry.size_bytes <= 0:
            raise ValueError("resume pack manifest is not contiguous")
        expected_offset += entry.size_bytes
    if not entries or expected_offset != manifest.size_bytes:
        raise ValueError("resume pack manifest size does not match its entries")
    return manifest


def _write_manifest(path: Path, manifest: ResumePackManifest) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_exact(
    source_fd: int,
    destination_fd: int,
    size_bytes: int,
    chunk_bytes: int,
    source_path: Path,
) -> None:
    remaining = size_bytes
    copy_file_range_available = hasattr(os, "copy_file_range")
    sendfile_available = hasattr(os, "sendfile")
    while remaining:
        size = min(chunk_bytes, remaining)
        copied = 0
        if copy_file_range_available:
            try:
                copied = os.copy_file_range(source_fd, destination_fd, size)
            except OSError as exc:
                if exc.errno not in {
                    errno.EXDEV,
                    errno.EINVAL,
                    errno.ENOSYS,
                    errno.EOPNOTSUPP,
                }:
                    raise
                copy_file_range_available = False
        if copied == 0 and sendfile_available:
            try:
                copied = os.sendfile(destination_fd, source_fd, None, size)
            except OSError as exc:
                if exc.errno not in {
                    errno.EINVAL,
                    errno.ENOSYS,
                    errno.EOPNOTSUPP,
                }:
                    raise
                sendfile_available = False
        if copied == 0 and not copy_file_range_available and not sendfile_available:
            content = os.read(source_fd, size)
            copied = len(content)
            written = 0
            while written < copied:
                written += os.write(destination_fd, content[written:])
        if copied <= 0:
            raise OSError(f"short copy while packing {source_path}")
        remaining -= copied


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
