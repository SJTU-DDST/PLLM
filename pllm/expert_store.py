from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .expert_catalog import ExpertCatalog


MAGIC = b"PLLMEX01"
HEADER = struct.Struct("<8sQ")
SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class PackedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(slots=True, frozen=True)
class ExpertPayload:
    layer: int
    expert: int
    format: str
    model_fingerprint: str
    tensors: tuple[PackedTensor, ...]
    data: bytes
    sha256: str

    @classmethod
    def create(
        cls,
        layer: int,
        expert: int,
        format: str,
        model_fingerprint: str,
        tensors: list[tuple[str, str, tuple[int, ...], bytes]],
    ) -> "ExpertPayload":
        packed: list[PackedTensor] = []
        chunks: list[bytes] = []
        offset = 0
        for name, dtype, shape, content in tensors:
            packed.append(
                PackedTensor(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    offset=offset,
                    size_bytes=len(content),
                )
            )
            chunks.append(content)
            offset += len(content)
        data = b"".join(chunks)
        return cls(
            layer=layer,
            expert=expert,
            format=format,
            model_fingerprint=model_fingerprint,
            tensors=tuple(packed),
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def tensor_bytes(self, name: str) -> bytes:
        tensor = next((item for item in self.tensors if item.name == name), None)
        if tensor is None:
            raise KeyError(name)
        end = tensor.offset + tensor.size_bytes
        return self.data[tensor.offset:end]


class ExpertSource(Protocol):
    def get(self, layer: int, expert: int) -> ExpertPayload: ...

    def contains(self, layer: int, expert: int) -> bool: ...


class ExpertPackageCodec:
    @staticmethod
    def encode(payload: ExpertPayload) -> bytes:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "layer": payload.layer,
            "expert": payload.expert,
            "format": payload.format,
            "model_fingerprint": payload.model_fingerprint,
            "sha256": payload.sha256,
            "size_bytes": len(payload.data),
            "tensors": [item.to_dict() for item in payload.tensors],
        }
        header = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        return HEADER.pack(MAGIC, len(header)) + header + payload.data

    @staticmethod
    def decode(blob: bytes, verify: bool = True) -> ExpertPayload:
        if len(blob) < HEADER.size:
            raise ValueError("expert package is shorter than its fixed header")
        magic, header_size = HEADER.unpack_from(blob)
        if magic != MAGIC:
            raise ValueError("expert package has an invalid magic")
        header_end = HEADER.size + header_size
        if header_size <= 0 or header_end > len(blob):
            raise ValueError("expert package has an invalid metadata length")
        try:
            metadata = json.loads(blob[HEADER.size:header_end])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("expert package metadata is invalid") from exc
        if not isinstance(metadata, dict):
            raise ValueError("expert package metadata must be an object")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported expert package schema")
        data = blob[header_end:]
        if int(metadata.get("size_bytes", -1)) != len(data):
            raise ValueError("expert package payload length mismatch")
        tensor_rows = metadata.get("tensors", [])
        if not isinstance(tensor_rows, list):
            raise ValueError("expert package tensor metadata must be a list")
        tensors = tuple(
            PackedTensor(
                name=str(item["name"]),
                dtype=str(item["dtype"]),
                shape=tuple(int(value) for value in item["shape"]),
                offset=int(item["offset"]),
                size_bytes=int(item["size_bytes"]),
            )
            for item in tensor_rows
        )
        _validate_tensor_layout(tensors, len(data))
        digest = hashlib.sha256(data).hexdigest()
        if verify and digest != metadata.get("sha256"):
            raise ValueError("expert package checksum mismatch")
        return ExpertPayload(
            layer=int(metadata["layer"]),
            expert=int(metadata["expert"]),
            format=str(metadata["format"]),
            model_fingerprint=str(metadata["model_fingerprint"]),
            tensors=tensors,
            data=data,
            sha256=digest,
        )

    @staticmethod
    def read(path: Path, verify: bool = True) -> ExpertPayload:
        return ExpertPackageCodec.decode(path.read_bytes(), verify=verify)


class SafetensorsExpertSource:
    """Reads one canonical expert with positional pread, never a whole shard."""

    def __init__(self, catalog: ExpertCatalog) -> None:
        self.catalog = catalog
        self.root = Path(catalog.model_path)
        self.model_fingerprint = model_fingerprint(self.root)
        self._objects = {
            (item.layer, item.expert): item for item in catalog.experts
        }

    def contains(self, layer: int, expert: int) -> bool:
        return (layer, expert) in self._objects

    def get(self, layer: int, expert: int) -> ExpertPayload:
        obj = self._objects.get((layer, expert))
        if obj is None:
            raise KeyError(f"unknown expert {layer}:{expert}")
        tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
        descriptors = sorted(obj.slices, key=lambda item: item.name)
        open_files: dict[str, int] = {}
        try:
            for item in descriptors:
                fd = open_files.get(item.shard)
                if fd is None:
                    fd = os.open(self.root / item.shard, os.O_RDONLY)
                    open_files[item.shard] = fd
                content = os.pread(fd, item.size_bytes, item.file_offset)
                if len(content) != item.size_bytes:
                    raise OSError(
                        f"short read for {item.name}: {len(content)}/{item.size_bytes}"
                    )
                tensors.append(
                    (item.name, item.dtype, tuple(item.shape), content)
                )
        finally:
            for fd in open_files.values():
                os.close(fd)
        return ExpertPayload.create(
            layer=layer,
            expert=expert,
            format="safetensors_raw_nvfp4",
            model_fingerprint=self.model_fingerprint,
            tensors=tensors,
        )


class SSDExpertStore:
    """Atomic, checksummed expert-object tier on local NVMe."""

    def __init__(
        self,
        root: str | Path,
        model_fingerprint: str,
        quota_bytes: int = 0,
        required_format: str = "vllm_runtime_nvfp4_marlin_v1",
    ) -> None:
        self.root = Path(root).expanduser()
        self.model_fingerprint = model_fingerprint
        self.quota_bytes = quota_bytes
        self.required_format = required_format
        self.root.mkdir(parents=True, exist_ok=True)
        self._puts_since_quota = 0
        self._cached_status: dict[str, Any] = {}
        self._status_at = 0.0

    def path_for(self, layer: int, expert: int) -> Path:
        return self.root / f"layer-{layer:03d}" / f"expert-{expert:04d}.pllmex"

    def contains(self, layer: int, expert: int) -> bool:
        return self.path_for(layer, expert).is_file()

    def get(self, layer: int, expert: int) -> ExpertPayload:
        path = self.path_for(layer, expert)
        payload = ExpertPackageCodec.read(path)
        self._validate(payload, layer, expert)
        try:
            os.utime(path, None)
        except OSError:
            pass
        return payload

    def put(self, payload: ExpertPayload) -> Path:
        self._validate(payload, payload.layer, payload.expert)
        destination = self.path_for(payload.layer, payload.expert)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = ExpertPackageCodec.encode(payload)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        self._puts_since_quota += 1
        self._cached_status = {}
        if self._puts_since_quota >= 256:
            self.enforce_quota(protected={destination})
            self._puts_since_quota = 0
        return destination

    def install_verified_package(
        self, package_path: Path, payload: ExpertPayload
    ) -> Path:
        """Atomically adopt a package already downloaded into this filesystem."""
        self._validate(payload, payload.layer, payload.expert)
        destination = self.path_for(payload.layer, payload.expert)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(package_path, destination)
        _fsync_directory(destination.parent)
        self._cached_status = {}
        self._puts_since_quota += 1
        if self._puts_since_quota >= 256:
            self.enforce_quota(protected={destination})
            self._puts_since_quota = 0
        return destination

    def status(self) -> dict[str, Any]:
        if self._cached_status and time.monotonic() - self._status_at < 30.0:
            return dict(self._cached_status)
        files = 0
        used = 0
        for path in self.root.glob("layer-*/expert-*.pllmex"):
            try:
                used += path.stat().st_size
                files += 1
            except OSError:
                continue
        self._cached_status = {
            "backend": "ssd_runtime_expert_store",
            "root": str(self.root),
            "format": self.required_format,
            "model_fingerprint": self.model_fingerprint,
            "objects": files,
            "used_bytes": used,
            "quota_bytes": self.quota_bytes,
        }
        self._status_at = time.monotonic()
        return dict(self._cached_status)

    def enforce_quota(self, protected: set[Path] | None = None) -> None:
        if self.quota_bytes <= 0:
            return
        protected = {item.resolve() for item in (protected or set())}
        candidates: list[tuple[float, int, Path]] = []
        total = 0
        for path in self.root.glob("layer-*/expert-*.pllmex"):
            try:
                stat = path.stat()
            except OSError:
                continue
            total += stat.st_size
            if path.resolve() not in protected:
                candidates.append((stat.st_atime, stat.st_size, path))
        for _atime, size, path in sorted(candidates):
            if total <= self.quota_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
        self._cached_status = {}

    def _validate(self, payload: ExpertPayload, layer: int, expert: int) -> None:
        if (payload.layer, payload.expert) != (layer, expert):
            raise ValueError("expert package key mismatch")
        if payload.model_fingerprint != self.model_fingerprint:
            raise ValueError("expert package belongs to another model checkpoint")
        if self.required_format and payload.format != self.required_format:
            raise ValueError(
                f"expected {self.required_format}, found {payload.format}"
            )


class RDMABridgeTransport:
    """Generic object get/put transport backed by pllm-rdma-store."""

    def __init__(
        self,
        peer: str,
        port: int,
        binary: str | Path,
        token_file: str | Path = "",
        timeout_seconds: float = 30.0,
        allocator: str = "cuda-host",
        device: str = "",
    ) -> None:
        if allocator not in {"aligned", "cuda-host"}:
            raise ValueError("RDMA allocator must be aligned or cuda-host")
        self.peer = peer
        self.port = port
        self.binary = Path(binary)
        self.token_file = Path(token_file).expanduser() if token_file else None
        self.timeout_seconds = timeout_seconds
        self.allocator = allocator
        self.device = device

    @property
    def available(self) -> bool:
        return bool(self.peer and self.binary.is_file())

    def get(self, key: str | Path, destination: Path) -> None:
        self._run("get", Path(key), destination)

    def put(self, key: str | Path, source: Path) -> None:
        self._run("put", Path(key), source)

    def _run(self, operation: str, key: Path, path: Path) -> None:
        command = [
            str(self.binary),
            "--client",
            self.peer,
            "--port",
            str(self.port),
            "--operation",
            operation,
            "--key",
            key.as_posix(),
            "--file",
            str(path),
            "--allocator",
            self.allocator,
        ]
        if self.device:
            command.extend(["--device", self.device])
        if self.token_file is not None:
            command.extend(["--token-file", str(self.token_file)])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise OSError(f"RDMA object {operation} failed: {error}")


class RDMAExpertStore:
    """Fills the local SSD tier through the host-staged verbs bridge."""

    def __init__(
        self,
        peer: str,
        port: int,
        binary: str | Path,
        local_cache: SSDExpertStore,
        token_file: str | Path = "",
        timeout_seconds: float = 30.0,
        allocator: str = "cuda-host",
        device: str = "",
    ) -> None:
        self.peer = peer
        self.port = port
        self.binary = Path(binary)
        self.local_cache = local_cache
        self.transport = RDMABridgeTransport(
            peer,
            port,
            binary,
            token_file=token_file,
            timeout_seconds=timeout_seconds,
            allocator=allocator,
            device=device,
        )

    def contains(self, layer: int, expert: int) -> bool:
        return self.transport.available

    def get(self, layer: int, expert: int) -> ExpertPayload:
        if self.local_cache.contains(layer, expert):
            try:
                return self.local_cache.get(layer, expert)
            except (OSError, ValueError):
                self.local_cache.path_for(layer, expert).unlink(missing_ok=True)
        relative = self.local_cache.path_for(layer, expert).relative_to(
            self.local_cache.root
        )
        destination = self.local_cache.path_for(layer, expert)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".rdma-", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            self._run("get", relative, temporary)
            payload = ExpertPackageCodec.read(temporary)
            self.local_cache._validate(payload, layer, expert)
            self.local_cache.install_verified_package(temporary, payload)
            return self.local_cache.get(layer, expert)
        finally:
            temporary.unlink(missing_ok=True)

    def put(self, payload: ExpertPayload) -> None:
        local_path = self.local_cache.put(payload)
        self.put_path(local_path, payload)

    def put_path(
        self, local_path: str | Path, payload: ExpertPayload | None = None
    ) -> None:
        local_path = Path(local_path)
        payload = payload or ExpertPackageCodec.read(local_path)
        self.local_cache._validate(payload, payload.layer, payload.expert)
        relative = local_path.relative_to(self.local_cache.root)
        self._run("put", relative, local_path)

    def _run(self, operation: str, key: Path, path: Path) -> None:
        if operation == "get":
            self.transport.get(key, path)
        else:
            self.transport.put(key, path)


class TieredExpertSource:
    def __init__(self, sources: list[ExpertSource]) -> None:
        self.sources = list(sources)

    def contains(self, layer: int, expert: int) -> bool:
        return any(source.contains(layer, expert) for source in self.sources)

    def get(self, layer: int, expert: int) -> ExpertPayload:
        errors: list[str] = []
        for source in self.sources:
            if not source.contains(layer, expert):
                continue
            try:
                return source.get(layer, expert)
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{source.__class__.__name__}: {exc}")
        detail = "; ".join(errors) if errors else "no configured tier contains it"
        raise FileNotFoundError(f"expert {layer}:{expert} unavailable: {detail}")


def model_fingerprint(model_root: str | Path) -> str:
    root = Path(model_root).expanduser().resolve()
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    for path in sorted(root.glob("*.safetensors")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()


def _validate_tensor_layout(tensors: tuple[PackedTensor, ...], size: int) -> None:
    occupied: list[tuple[int, int]] = []
    names: set[str] = set()
    for tensor in tensors:
        if tensor.name in names:
            raise ValueError(f"duplicate packed tensor {tensor.name}")
        names.add(tensor.name)
        end = tensor.offset + tensor.size_bytes
        if tensor.offset < 0 or tensor.size_bytes < 0 or end > size:
            raise ValueError(f"packed tensor {tensor.name} exceeds payload")
        occupied.append((tensor.offset, end))
    for previous, current in zip(sorted(occupied), sorted(occupied)[1:]):
        if previous[1] > current[0]:
            raise ValueError("packed tensor regions overlap")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
