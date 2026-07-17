from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .expert_store import RDMABridgeTransport


SCHEMA_VERSION = 1
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(slots=True, frozen=True)
class StateComponent:
    name: str
    data: bytes
    dtype: str = "bytes"
    shape: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class HiberStateSnapshot:
    request_id: str
    epoch: int
    model_fingerprint: str
    committed_tokens: int
    components: tuple[StateComponent, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class HiberStateStore:
    """Transactional live-request state store for SSD and remote RDMA tiers."""

    def __init__(
        self,
        root: str | Path,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        transport: RDMABridgeTransport | None = None,
        remote_prefix: str = "",
    ) -> None:
        if chunk_bytes <= 0 or chunk_bytes > 128 * 1024 * 1024:
            raise ValueError("hiber-state chunks must be within (0, 128 MiB]")
        self.root = Path(root).expanduser()
        self.chunk_bytes = chunk_bytes
        self.transport = transport
        self.remote_prefix = Path(remote_prefix) if remote_prefix else Path()

    def commit(
        self, snapshot: HiberStateSnapshot, replicate: bool = False
    ) -> dict[str, Any]:
        self._validate_snapshot(snapshot)
        request_root = self.root / snapshot.request_id
        request_root.mkdir(parents=True, exist_ok=True)
        destination = request_root / f"epoch-{snapshot.epoch:020d}"
        if destination.exists():
            raise FileExistsError(f"hiber-state epoch already exists: {destination}")
        temporary = request_root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(mode=0o700)
        try:
            components = []
            for component in snapshot.components:
                chunks = []
                digest = hashlib.sha256()
                for index, offset in enumerate(
                    range(0, len(component.data), self.chunk_bytes)
                ):
                    content = component.data[offset : offset + self.chunk_bytes]
                    name = f"{component.name}.{index:06d}.chunk"
                    _write_durable(temporary / name, content)
                    digest.update(content)
                    chunks.append(
                        {
                            "name": name,
                            "size_bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    )
                components.append(
                    {
                        "name": component.name,
                        "dtype": component.dtype,
                        "shape": list(component.shape),
                        "size_bytes": len(component.data),
                        "sha256": digest.hexdigest(),
                        "metadata": component.metadata,
                        "chunks": chunks,
                    }
                )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "state": "committed",
                "request_id": snapshot.request_id,
                "epoch": snapshot.epoch,
                "model_fingerprint": snapshot.model_fingerprint,
                "committed_tokens": snapshot.committed_tokens,
                "created_at": time.time(),
                "metadata": snapshot.metadata,
                "components": components,
            }
            manifest_bytes = (
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            _write_durable(temporary / "commit.json", manifest_bytes)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(request_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        remote_committed = False
        if replicate:
            self.replicate(snapshot.request_id, snapshot.epoch)
            remote_committed = True
        return {
            "request_id": snapshot.request_id,
            "epoch": snapshot.epoch,
            "committed_tokens": snapshot.committed_tokens,
            "path": str(destination),
            "components": len(snapshot.components),
            "remote_committed": remote_committed,
        }

    def load(
        self,
        request_id: str,
        epoch: int | None = None,
        expected_model_fingerprint: str = "",
    ) -> HiberStateSnapshot:
        _validate_name(request_id, "request_id")
        transaction = self._transaction_path(request_id, epoch)
        return self._decode_transaction(
            transaction, request_id, expected_model_fingerprint
        )

    def _decode_transaction(
        self,
        transaction: Path,
        request_id: str,
        expected_model_fingerprint: str,
    ) -> HiberStateSnapshot:
        manifest = self._read_manifest(transaction / "commit.json")
        self._validate_manifest(
            manifest,
            request_id,
            int(manifest.get("epoch", -1)),
            expected_model_fingerprint,
        )
        components: list[StateComponent] = []
        for row in manifest["components"]:
            chunks = []
            component_digest = hashlib.sha256()
            for chunk in row["chunks"]:
                path = transaction / str(chunk["name"])
                content = path.read_bytes()
                if len(content) != int(chunk["size_bytes"]):
                    raise ValueError(f"hiber-state chunk size mismatch: {path.name}")
                if hashlib.sha256(content).hexdigest() != chunk["sha256"]:
                    raise ValueError(f"hiber-state chunk checksum mismatch: {path.name}")
                component_digest.update(content)
                chunks.append(content)
            data = b"".join(chunks)
            if len(data) != int(row["size_bytes"]):
                raise ValueError(f"hiber-state component size mismatch: {row['name']}")
            if component_digest.hexdigest() != row["sha256"]:
                raise ValueError(
                    f"hiber-state component checksum mismatch: {row['name']}"
                )
            components.append(
                StateComponent(
                    name=str(row["name"]),
                    data=data,
                    dtype=str(row.get("dtype", "bytes")),
                    shape=tuple(int(item) for item in row.get("shape", [])),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        try:
            os.utime(transaction, None)
        except OSError:
            pass
        return HiberStateSnapshot(
            request_id=request_id,
            epoch=int(manifest["epoch"]),
            model_fingerprint=str(manifest["model_fingerprint"]),
            committed_tokens=int(manifest["committed_tokens"]),
            components=tuple(components),
            metadata=dict(manifest.get("metadata") or {}),
        )

    def replicate(self, request_id: str, epoch: int) -> None:
        if self.transport is None or not self.transport.available:
            raise RuntimeError("RDMA hiber-state transport is unavailable")
        transaction = self._transaction_path(request_id, epoch)
        manifest = self._read_manifest(transaction / "commit.json")
        self._validate_manifest(manifest, request_id, epoch, "")
        for row in manifest["components"]:
            component_digest = hashlib.sha256()
            component_size = 0
            for chunk in row["chunks"]:
                path = transaction / str(chunk["name"])
                content = path.read_bytes()
                if len(content) != int(chunk["size_bytes"]):
                    raise ValueError(f"hiber-state chunk size mismatch: {path.name}")
                if hashlib.sha256(content).hexdigest() != chunk["sha256"]:
                    raise ValueError(f"hiber-state chunk checksum mismatch: {path.name}")
                component_digest.update(content)
                component_size += len(content)
                self.transport.put(self._remote_key(request_id, epoch, path.name), path)
            if component_size != int(row["size_bytes"]):
                raise ValueError(f"hiber-state component size mismatch: {row['name']}")
            if component_digest.hexdigest() != row["sha256"]:
                raise ValueError(
                    f"hiber-state component checksum mismatch: {row['name']}"
                )
        self.transport.put(
            self._remote_key(request_id, epoch, "commit.json"),
            transaction / "commit.json",
        )

    def fetch_remote(
        self,
        request_id: str,
        epoch: int,
        expected_model_fingerprint: str = "",
    ) -> HiberStateSnapshot:
        _validate_name(request_id, "request_id")
        if self.transport is None or not self.transport.available:
            raise RuntimeError("RDMA hiber-state transport is unavailable")
        request_root = self.root / request_id
        request_root.mkdir(parents=True, exist_ok=True)
        destination = request_root / f"epoch-{epoch:020d}"
        if destination.exists():
            return self.load(request_id, epoch, expected_model_fingerprint)
        temporary = request_root / f".{destination.name}.{uuid.uuid4().hex}.fetch"
        temporary.mkdir(mode=0o700)
        try:
            manifest_path = temporary / "commit.json"
            self.transport.get(
                self._remote_key(request_id, epoch, "commit.json"), manifest_path
            )
            manifest = self._read_manifest(manifest_path)
            self._validate_manifest(
                manifest, request_id, epoch, expected_model_fingerprint
            )
            for row in manifest["components"]:
                for chunk in row["chunks"]:
                    name = str(chunk["name"])
                    self.transport.get(
                        self._remote_key(request_id, epoch, name), temporary / name
                    )
            self._decode_transaction(
                temporary, request_id, expected_model_fingerprint
            )
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(request_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.load(request_id, epoch, expected_model_fingerprint)

    def status(self) -> dict[str, Any]:
        transactions = 0
        used = 0
        newest = 0.0
        for path in self.root.glob("*/epoch-*"):
            if not (path / "commit.json").is_file():
                continue
            transactions += 1
            try:
                newest = max(newest, path.stat().st_mtime)
                used += sum(
                    item.stat().st_size for item in path.iterdir() if item.is_file()
                )
            except OSError:
                continue
        return {
            "backend": "transactional_ssd_hiber_state",
            "root": str(self.root),
            "transactions": transactions,
            "used_bytes": used,
            "chunk_bytes": self.chunk_bytes,
            "remote_enabled": bool(self.transport and self.transport.available),
            "newest_at": newest or None,
            "serializer_attached": False,
        }

    def transaction_entries(self) -> list[tuple[float, int, Path]]:
        entries = []
        for path in self.root.glob("*/epoch-*"):
            if not (path / "commit.json").is_file():
                continue
            try:
                size = sum(
                    item.stat().st_size for item in path.iterdir() if item.is_file()
                )
                entries.append((path.stat().st_atime, size, path))
            except OSError:
                continue
        return entries

    def _transaction_path(self, request_id: str, epoch: int | None) -> Path:
        _validate_name(request_id, "request_id")
        request_root = self.root / request_id
        if epoch is None:
            candidates = sorted(request_root.glob("epoch-*"))
            if not candidates:
                raise FileNotFoundError(f"no hiber-state for request {request_id}")
            return candidates[-1]
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        return request_root / f"epoch-{epoch:020d}"

    def _remote_key(self, request_id: str, epoch: int, name: str) -> Path:
        return self.remote_prefix / request_id / f"epoch-{epoch:020d}" / name

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid hiber-state manifest: {path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("hiber-state manifest must be an object")
        return manifest

    @staticmethod
    def _validate_manifest(
        manifest: dict[str, Any],
        request_id: str,
        epoch: int,
        expected_model_fingerprint: str,
    ) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported hiber-state schema")
        if manifest.get("state") != "committed":
            raise ValueError("hiber-state transaction is not committed")
        if not manifest.get("model_fingerprint"):
            raise ValueError("hiber-state model fingerprint is missing")
        if int(manifest.get("committed_tokens", -1)) < 0:
            raise ValueError("hiber-state committed token count is invalid")
        if manifest.get("request_id") != request_id or int(
            manifest.get("epoch", -1)
        ) != epoch:
            raise ValueError("hiber-state transaction identity mismatch")
        if expected_model_fingerprint and manifest.get(
            "model_fingerprint"
        ) != expected_model_fingerprint:
            raise ValueError("hiber-state belongs to another model checkpoint")
        rows = manifest.get("components")
        if not isinstance(rows, list) or not rows:
            raise ValueError("hiber-state manifest has no components")
        names = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("hiber-state component metadata is invalid")
            name = str(row.get("name", ""))
            _validate_name(name, "component name")
            if name in names:
                raise ValueError(f"duplicate hiber-state component: {name}")
            names.add(name)
            chunks = row.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise ValueError(f"hiber-state component has no chunks: {name}")
            if int(row.get("size_bytes", 0)) <= 0:
                raise ValueError(f"hiber-state component size is invalid: {name}")
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise ValueError("hiber-state chunk metadata is invalid")
                chunk_name = str(chunk.get("name", ""))
                _validate_name(chunk_name, "chunk name")
                chunk_size = int(chunk.get("size_bytes", 0))
                if chunk_size <= 0 or chunk_size > 128 * 1024 * 1024:
                    raise ValueError(f"hiber-state chunk size is invalid: {chunk_name}")

    @staticmethod
    def _validate_snapshot(snapshot: HiberStateSnapshot) -> None:
        _validate_name(snapshot.request_id, "request_id")
        if snapshot.epoch < 0 or snapshot.committed_tokens < 0:
            raise ValueError("epoch and committed_tokens must be non-negative")
        if not snapshot.model_fingerprint:
            raise ValueError("model_fingerprint is required")
        if not snapshot.components:
            raise ValueError("at least one live-state component is required")
        names = set()
        for component in snapshot.components:
            _validate_name(component.name, "component name")
            if component.name in names:
                raise ValueError(f"duplicate live-state component: {component.name}")
            if not component.data:
                raise ValueError(f"live-state component is empty: {component.name}")
            names.add(component.name)
        # Fail before writing if caller metadata is not JSON serializable.
        json.dumps(snapshot.metadata, ensure_ascii=False)
        for component in snapshot.components:
            json.dumps(component.metadata, ensure_ascii=False)


def _validate_name(value: str, label: str) -> None:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")


def _write_durable(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(content)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short hiber-state write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
