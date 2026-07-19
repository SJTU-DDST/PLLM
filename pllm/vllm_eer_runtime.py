from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_EXPERT_CACHE_DIR, pllm_runtime_dir
from .expert_dataplane import ExpertSlotDataPlane
from .expert_catalog import ExpertCatalog
from .decode_residency import DecodeRouteWindow
from .expert_store import (
    ExpertPayload,
    RDMAExpertStore,
    RDMAPoolExpertStore,
    RDMAPoolStream,
    SSDExpertStore,
    TieredExpertSource,
    model_fingerprint,
)


SUPPORTED_VLLM = "0.25.1"
RUNTIME_FORMAT = "vllm_runtime_nvfp4_marlin_v1"
LAYER_PATTERN = re.compile(r"layers\.(?P<layer>\d+)\.mixer\.experts$")
RUNTIME_TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
)


def low_memory_nvfp4_scale_factor(
    marlin_scales: Any, a_dtype: Any | None = None
) -> float:
    """Compute vLLM's exact NVFP4 scale factor without a full FP32 copy."""
    import torch

    if a_dtype is not None and a_dtype == torch.half:
        return 1.0
    max_scale = marlin_scales.amax()
    if bool(max_scale > 0):
        max_val = max_scale.float() * (2**7)
        if bool(max_val < 448 * (2**7)):
            return (448 * (2**7) / max_val).log2().floor().exp2().item()
    return 1.0


def tensor_storage_bytes(tensor: Any) -> bytes:
    """Return contiguous tensor storage bytes, including scalar tensors."""
    import torch

    flat = tensor.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8).cpu().numpy().tobytes()


def release_export_layer(layer: Any) -> None:
    """Drop a persisted export layer so conversion peak is layer-bounded."""
    import torch

    method = layer.quant_method
    method.moe_kernel = None
    method.moe_quant_config = None
    for name in RUNTIME_TENSORS:
        if name in layer._parameters:
            parameter = layer._parameters.pop(name)
            parameter.grad = None
    if hasattr(layer, "workspace"):
        delattr(layer, "workspace")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def flatten_cache_tensors(value: Any) -> list[Any]:
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return [] if value is None else [value]
    flattened: list[Any] = []
    for item in items:
        flattened.extend(flatten_cache_tensors(item))
    return flattened


def cache_storage_signature(
    value: Any, *, sample_content: bool = False, sample_bytes: int = 64
) -> dict[str, Any]:
    """Describe cache allocations and optionally fingerprint tiny content samples."""
    import torch

    if sample_bytes <= 0:
        raise ValueError("sample_bytes must be positive")
    storages: dict[tuple[str, int], int] = {}
    representatives: dict[tuple[str, int], Any] = {}
    tensors = flatten_cache_tensors(value)
    for tensor in tensors:
        if not hasattr(tensor, "untyped_storage"):
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()))
        storages[key] = max(storages.get(key, 0), int(storage.nbytes()))
        representatives.setdefault(key, tensor)
    encoded = "|".join(
        f"{device}:{pointer}:{size}"
        for (device, pointer), size in sorted(storages.items())
    ).encode()
    content = hashlib.sha256()
    sampled = 0
    if sample_content:
        for key in sorted(representatives):
            tensor = representatives[key].detach()
            if not tensor.is_contiguous():
                raise RuntimeError("state-island sampling requires contiguous cache tensors")
            raw = tensor.view(torch.uint8).reshape(-1)
            length = int(raw.numel())
            if length <= 0:
                continue
            width = min(sample_bytes, length)
            offsets = sorted({0, max(0, (length - width) // 2), length - width})
            content.update(f"{key[0]}:{key[1]}:{length}".encode())
            for offset in offsets:
                payload = raw[offset : offset + width].cpu().numpy().tobytes()
                content.update(payload)
                sampled += len(payload)
    return {
        "attached": bool(tensors),
        "tensor_count": len(tensors),
        "storage_count": len(storages),
        "allocated_bytes": sum(storages.values()),
        "allocation_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "content_sample_fingerprint": content.hexdigest() if sample_content else None,
        "sampled_content_bytes": sampled,
        "copy_bytes": sampled,
        "scope": "attention_kv_and_mamba_conv_ssm_allocations",
    }


@dataclass(slots=True, frozen=True)
class EERRuntimeConfig:
    mode: str
    slots_per_layer: int
    model_path: Path
    cache_dir: Path
    cache_quota_bytes: int
    socket_path: Path
    rdma_peer: str
    rdma_port: int
    rdma_binary: Path
    rdma_token_file: Path | None
    rdma_allocator: str
    rdma_device: str
    rdma_ib_port: int
    rdma_gid_index: int
    rdma_pool_port: int
    rdma_pool_binary: Path
    rdma_pool_index: Path | None
    rdma_cuda_register_staging: bool
    route_window_steps: int

    @classmethod
    def from_environment(cls) -> "EERRuntimeConfig":
        mode = os.getenv("PLLM_EER_MODE", "off").strip().lower()
        if mode not in {"off", "export", "elastic"}:
            raise ValueError("PLLM_EER_MODE must be off, export, or elastic")
        runtime_dir = pllm_runtime_dir()
        return cls(
            mode=mode,
            slots_per_layer=int(os.getenv("PLLM_EER_SLOTS_PER_LAYER", "128")),
            model_path=Path(
                os.getenv(
                    "MODEL_PATH",
                    "/mnt/ssd-storage/shared_models/"
                    "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                )
            ).expanduser(),
            cache_dir=Path(
                os.getenv("PLLM_EER_CACHE_DIR", str(DEFAULT_EXPERT_CACHE_DIR))
            ).expanduser(),
            cache_quota_bytes=int(
                float(os.getenv("PLLM_EER_CACHE_QUOTA_GIB", "80")) * 1024**3
            ),
            socket_path=Path(
                os.getenv("PLLM_EER_SOCKET", str(runtime_dir / "pllm-eer.sock"))
            ),
            rdma_peer=os.getenv("PLLM_EER_RDMA_PEER", "").strip(),
            rdma_port=int(os.getenv("PLLM_EER_RDMA_PORT", "17900")),
            rdma_binary=Path(
                os.getenv(
                    "PLLM_EER_RDMA_BINARY",
                    str(Path.cwd() / "rdma_bridge/build/pllm-rdma-store"),
                )
            ),
            rdma_token_file=(
                Path(os.environ["PLLM_EER_RDMA_TOKEN_FILE"]).expanduser()
                if os.getenv("PLLM_EER_RDMA_TOKEN_FILE")
                else None
            ),
            rdma_allocator=os.getenv("PLLM_EER_RDMA_ALLOCATOR", "cuda-host"),
            rdma_device=os.getenv("PLLM_EER_RDMA_DEVICE", "").strip(),
            rdma_ib_port=int(os.getenv("PLLM_EER_RDMA_IB_PORT", "1")),
            rdma_gid_index=int(os.getenv("PLLM_EER_RDMA_GID_INDEX", "0")),
            rdma_pool_port=int(os.getenv("PLLM_EER_RDMA_POOL_PORT", "17902")),
            rdma_pool_binary=Path(
                os.getenv(
                    "PLLM_EER_RDMA_POOL_BINARY",
                    str(Path.cwd() / "rdma_bridge/build/pllm-rdma-pool"),
                )
            ),
            rdma_pool_index=(
                Path(os.environ["PLLM_EER_RDMA_POOL_INDEX"]).expanduser()
                if os.getenv("PLLM_EER_RDMA_POOL_INDEX")
                else None
            ),
            rdma_cuda_register_staging=(
                os.getenv("PLLM_EER_RDMA_CUDA_REGISTER_STAGING", "1") == "1"
            ),
            route_window_steps=int(os.getenv("PLLM_EER_ROUTE_WINDOW_STEPS", "256")),
        )


class TorchMarlinSlotSink:
    required_format = RUNTIME_FORMAT

    def __init__(self, layer: Any, layer_id: int, fingerprint: str) -> None:
        import torch

        self.torch = torch
        self.layer = layer
        self.layer_id = layer_id
        self.fingerprint = fingerprint
        self.slot_count = int(layer.num_experts)
        self._specs = self._tensor_specs()

    def export(self, logical_expert: int, physical_slot: int) -> ExpertPayload:
        tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
        for name, dtype, shape, _device in self._specs:
            tensor = getattr(self.layer, name).data[physical_slot]
            tensors.append(
                (
                    name,
                    _dtype_name(dtype),
                    tuple(shape),
                    tensor_storage_bytes(tensor),
                )
            )
        return ExpertPayload.create(
            layer=self.layer_id,
            expert=logical_expert,
            format=RUNTIME_FORMAT,
            model_fingerprint=self.fingerprint,
            tensors=tensors,
        )

    def write(self, slot: int, payload: ExpertPayload) -> None:
        expected = {name: (dtype, shape) for name, dtype, shape, _ in self._specs}
        present = {item.name for item in payload.tensors}
        if present != set(expected):
            missing = sorted(set(expected) - present)
            extra = sorted(present - set(expected))
            raise ValueError(
                f"runtime expert tensor set mismatch; missing={missing}, extra={extra}"
            )
        for item in payload.tensors:
            expected_dtype, expected_shape = expected[item.name]
            if _dtype_name(expected_dtype) != item.dtype or expected_shape != item.shape:
                raise ValueError(f"runtime tensor layout mismatch for {item.name}")
            raw = payload.tensor_bytes(item.name)
            cpu_bytes = self.torch.frombuffer(raw, dtype=self.torch.uint8)
            cpu_tensor = cpu_bytes.view(expected_dtype).reshape(expected_shape)
            target = getattr(self.layer, item.name).data[slot]
            target.copy_(cpu_tensor.to(device=target.device), non_blocking=False)

    def invalidate(self, slot: int) -> None:
        # Bytes are immutable and inaccessible once expert_map is unpublished.
        # Clearing a multi-MiB slot would add latency without releasing capacity.
        del slot

    def publish_mapping(self, mapping: dict[int, int], generation: int) -> None:
        target = self.layer._expert_map
        host = self.torch.full(
            (self.layer.global_num_experts,), -1, dtype=self.torch.int32
        )
        for logical, physical in mapping.items():
            host[logical] = physical
        target.copy_(host.to(device=target.device), non_blocking=False)
        self.layer._pllm_eer_logical_to_slot = dict(mapping)
        self.layer._pllm_eer_generation = generation

    def begin_resize(self, slot_count: int) -> None:
        torch = self.torch
        method = self.layer.quant_method
        method.moe_kernel = None
        method.moe_quant_config = None
        specs = self._tensor_specs()
        for name, _dtype, _shape, _device in specs:
            delattr(self.layer, name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for name, dtype, row_shape, device in specs:
            parameter = torch.nn.Parameter(
                torch.empty((slot_count, *row_shape), dtype=dtype, device=device),
                requires_grad=False,
            )
            self.layer.register_parameter(name, parameter)
        self.slot_count = slot_count
        self.layer.num_experts = slot_count
        self.layer.local_num_experts = slot_count
        self.layer.moe_config.num_local_experts = slot_count
        self._specs = self._tensor_specs()
        self.publish_mapping({}, getattr(self.layer, "_pllm_eer_generation", 0) + 1)

    def resize_with_retained(
        self, slot_count: int, retained: list[tuple[int, int]]
    ) -> dict[str, Any]:
        torch = self.torch
        method = self.layer.quant_method
        method.moe_kernel = None
        method.moe_quant_config = None
        specs = self._tensor_specs()
        old_parameters = {name: getattr(self.layer, name) for name, *_ in specs}
        new_parameters: dict[str, Any] = {}
        bytes_copied = 0
        old_slots = torch.tensor(
            [physical for _logical, physical in retained],
            dtype=torch.long,
            device=next(iter(old_parameters.values())).device,
        )
        for name, dtype, row_shape, device in specs:
            parameter = torch.nn.Parameter(
                torch.empty((slot_count, *row_shape), dtype=dtype, device=device),
                requires_grad=False,
            )
            old = old_parameters[name].data
            for offset in range(0, len(retained), 8):
                indexes = old_slots[offset : offset + 8]
                parameter.data[offset : offset + len(indexes)].copy_(
                    old.index_select(0, indexes), non_blocking=False
                )
            bytes_copied += len(retained) * old[0].numel() * old.element_size()
            new_parameters[name] = parameter
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for name, parameter in new_parameters.items():
            setattr(self.layer, name, parameter)
        old_parameters.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.slot_count = slot_count
        self.layer.num_experts = slot_count
        self.layer.local_num_experts = slot_count
        self.layer.moe_config.num_local_experts = slot_count
        self._specs = self._tensor_specs()
        mapping = {logical: index for index, (logical, _old) in enumerate(retained)}
        self.publish_mapping(mapping, getattr(self.layer, "_pllm_eer_generation", 0) + 1)
        return {"mapping": mapping, "bytes_copied": bytes_copied}

    def finish_resize(self) -> None:
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            make_nvfp4_moe_kernel,
        )

        method = self.layer.quant_method
        method.moe_quant_config = method.get_fused_moe_quant_config(self.layer)
        method.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=method.moe_quant_config,
            moe_config=method.moe,
            experts_cls=method.experts_cls,
            backend=method.nvfp4_backend,
            routing_tables=self.layer._expert_routing_tables(),
            layer=self.layer,
        )
        method.moe_kernel.fused_experts.process_weights_after_loading(self.layer)

    def _tensor_specs(self) -> list[tuple[str, Any, tuple[int, ...], Any]]:
        specs = []
        for name in RUNTIME_TENSORS:
            tensor = getattr(self.layer, name, None)
            if tensor is None or tensor.ndim == 0:
                continue
            if tensor.shape[0] != self.slot_count:
                continue
            specs.append((name, tensor.dtype, tuple(tensor.shape[1:]), tensor.device))
        if not {"w13_weight", "w2_weight"}.issubset(item[0] for item in specs):
            raise RuntimeError("Marlin expert tensors are not available on the layer")
        return specs


class EERRuntime:
    def __init__(self, config: EERRuntimeConfig) -> None:
        self.config = config
        self.fingerprint = model_fingerprint(config.model_path)
        catalog = ExpertCatalog.from_model(config.model_path)
        self.expected_objects = len(catalog.experts)
        self.expected_layers = len(catalog.moe_layers)
        self.route_window = DecodeRouteWindow(
            catalog.moe_layers,
            catalog.experts_per_layer,
            config.route_window_steps,
        )
        self._last_moe_layer = max(catalog.moe_layers)
        self._miss_debt_budget_ms = float(
            os.getenv("PLLM_EER_MISS_DEBT_BUDGET_MS", "400")
        )
        self._miss_debt_ms = 0.0
        self._miss_debt_load_ms = 0.0
        self._miss_debt_tokens = 0
        self._miss_debt_violations = 0
        self._miss_debt_exceeded = False
        self._current_token_load_ms = 0.0
        self._miss_debt_lock = threading.Lock()
        self.local_store = SSDExpertStore(
            config.cache_dir,
            model_fingerprint=self.fingerprint,
            quota_bytes=config.cache_quota_bytes,
            required_format=RUNTIME_FORMAT,
        )
        remote_sources: list[Any] = []
        remote_store = None
        self.remote_pool: RDMAPoolExpertStore | None = None
        if config.rdma_peer:
            if config.rdma_token_file is None or not config.rdma_token_file.is_file():
                raise RuntimeError("RDMA warm source requires PLLM_EER_RDMA_TOKEN_FILE")
            if config.rdma_pool_index is not None:
                stream = RDMAPoolStream(
                    peer=config.rdma_peer,
                    port=config.rdma_pool_port,
                    binary=config.rdma_pool_binary,
                    index_file=config.rdma_pool_index,
                    token_file=config.rdma_token_file,
                    allocator=config.rdma_allocator,
                    device=config.rdma_device,
                    ib_port=config.rdma_ib_port,
                    gid_index=config.rdma_gid_index,
                    shared_staging=True,
                    cuda_register_staging=config.rdma_cuda_register_staging,
                )
                self.remote_pool = RDMAPoolExpertStore(stream, self.local_store)
                remote_sources.append(self.remote_pool)
                threading.Thread(
                    target=self.remote_pool.prime,
                    name="pllm-rdma-pool-prime",
                    daemon=True,
                ).start()
            else:
                remote_store = RDMAExpertStore(
                    peer=config.rdma_peer,
                    port=config.rdma_port,
                    binary=config.rdma_binary,
                    local_cache=self.local_store,
                    token_file=config.rdma_token_file or "",
                    allocator=config.rdma_allocator,
                    device=config.rdma_device,
                    ib_port=config.rdma_ib_port,
                    gid_index=config.rdma_gid_index,
                )
                remote_sources.append(remote_store)
        if (
            config.mode == "elastic"
            and not self._export_manifest_valid()
            and remote_store is not None
        ):
            self._fetch_remote_manifest(remote_store)
        if config.mode == "elastic" and not self._export_manifest_valid():
            raise RuntimeError(
                "elastic EER requires a complete local or RDMA runtime manifest"
            )
        self.initial_source = TieredExpertSource(
            [self.local_store, *remote_sources]
        )
        self.runtime_source = TieredExpertSource(
            [*remote_sources, self.local_store]
        )
        self.data_plane = ExpertSlotDataPlane(self.initial_source)
        self.sinks: dict[int, TorchMarlinSlotSink] = {}
        self.exported_objects = 0
        self.started_at = time.time()
        self.last_error = ""
        self.suspended = False
        self.transitioning = False
        self.faulted = False
        self._command_lock = threading.RLock()
        self._control_server: _ThreadedUnixServer | None = None
        self._control_thread: threading.Thread | None = None
        self._loader_cache_released = False
        self._model_runner: Any | None = None
        self._last_state_island_guard: dict[str, Any] = {
            "checked": False,
            "preserved": None,
        }

    def bind_model_runner(self, model_runner: Any) -> None:
        self._model_runner = model_runner

    def state_island_status(self, *, sample_content: bool = False) -> dict[str, Any]:
        if self._model_runner is None:
            return {
                "attached": False,
                "allocated_bytes": 0,
                "copy_bytes": 0,
                "scope": "attention_kv_and_mamba_conv_ssm_allocations",
                "resize_guard": dict(self._last_state_island_guard),
            }
        result = cache_storage_signature(
            getattr(self._model_runner, "kv_caches", None),
            sample_content=sample_content,
        )
        result["resize_guard"] = dict(self._last_state_island_guard)
        return result

    def release_loader_cache(self) -> None:
        if self._loader_cache_released:
            return
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loader_cache_released = True

    def start_control_server(self) -> None:
        if self.config.mode == "off" or self._control_server is not None:
            return
        self.config.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.socket_path.unlink(missing_ok=True)
        server = _ThreadedUnixServer(str(self.config.socket_path), _ControlHandler)
        server.runtime = self
        self._control_server = server
        self._control_thread = threading.Thread(
            target=server.serve_forever, name="pllm-eer-control", daemon=True
        )
        self._control_thread.start()

    def initialize_layer(self, layer: Any) -> None:
        from vllm.config import get_current_vllm_config

        layer_id = _layer_id(layer)
        if self.config.mode == "elastic":
            current = get_current_vllm_config()
            if current.model_config is not None and not current.model_config.enforce_eager:
                raise RuntimeError("PLLM EER requires vLLM --enforce-eager")
            backend = str(layer.quant_method.nvfp4_backend.value).lower()
            if backend != "marlin":
                raise RuntimeError(f"PLLM EER requires the Marlin MoE backend, found {backend}")
            if layer.quant_method.is_monolithic:
                raise RuntimeError("PLLM EER requires a modular MoE kernel with visible Top-k IDs")
        self.start_control_server()
        sink = TorchMarlinSlotSink(layer, layer_id, self.fingerprint)
        if self.config.mode == "export":
            mapping = getattr(layer, "_pllm_eer_logical_to_slot", None)
            if not mapping:
                mapping = {index: index for index in range(layer.num_experts)}
            for logical, physical in mapping.items():
                self.local_store.put(sink.export(logical, physical))
                self.exported_objects += 1
            if self.exported_objects >= self.expected_objects:
                self._write_export_manifest()
            release_export_layer(layer)
            return
        if self.config.mode != "elastic":
            return
        if layer_id in self.sinks:
            self.data_plane.unregister_layer(layer_id)
        self.sinks[layer_id] = sink
        initial = list(range(min(sink.slot_count, layer.global_num_experts)))
        self.data_plane.register_layer(
            layer=layer_id,
            global_experts=layer.global_num_experts,
            sink=sink,
            initial_experts=initial,
        )
        if len(self.sinks) == self.expected_layers:
            self.data_plane.source = self.runtime_source

    def ensure(self, layer: Any, topk_ids: Any) -> None:
        if self.config.mode != "elastic":
            return
        if self.suspended:
            raise RuntimeError("PLLM EER data plane is suspended")
        layer_id = _layer_id(layer)
        phase = self.route_window.current_phase()
        fully_resident = self.data_plane.is_fully_resident(layer_id)
        if fully_resident and phase != "decode":
            return
        if not fully_resident and phase != "decode":
            raise RuntimeError(
                "a non-full expert profile may execute only in decode; "
                "new prefill must be admitted through the PLLM proxy"
            )
        detached = topk_ids.detach()
        if detached.ndim == 1:
            detached = detached.reshape(1, -1)
        else:
            detached = detached.reshape(-1, detached.shape[-1])
        rows = detached.cpu().tolist()
        self.route_window.observe_rows(layer_id, rows)
        experts = list(
            dict.fromkeys(
                int(item) for row in rows for item in row if int(item) >= 0
            )
        )
        if fully_resident:
            return
        load_started = time.perf_counter()
        self.data_plane.ensure(layer_id, experts, reason="actual_topk")
        load_ms = (time.perf_counter() - load_started) * 1000.0
        self._record_miss_load(layer_id, len(rows), load_ms)

    def _record_miss_load(
        self, layer_id: int, token_rows: int, load_ms: float
    ) -> None:
        with self._miss_debt_lock:
            self._current_token_load_ms += load_ms
            if layer_id == self._last_moe_layer:
                tokens = max(1, int(token_rows))
                self._miss_debt_load_ms += self._current_token_load_ms
                self._miss_debt_tokens += tokens
                allowance = self._miss_debt_budget_ms * tokens
                self._miss_debt_ms = max(
                    0.0,
                    self._miss_debt_ms + self._current_token_load_ms - allowance,
                )
                if self._miss_debt_ms > self._miss_debt_budget_ms:
                    self._miss_debt_exceeded = True
                    self._miss_debt_violations += 1
                self._current_token_load_ms = 0.0

    def handle_command(self, request: dict[str, Any]) -> dict[str, Any]:
        command = str(request.get("command", "status"))
        if command == "status":
            return self.status()
        if self.config.mode != "elastic":
            raise RuntimeError("expert data plane is not in elastic mode")
        with self._command_lock:
            self.transitioning = True
            try:
                result = self._handle_mutating_command(command, request)
                if command == "resize":
                    self.faulted = False
                    self.last_error = ""
                return result
            except Exception:
                if command in {"resize", "suspend"}:
                    self.faulted = True
                raise
            finally:
                self.transitioning = False

    def _handle_mutating_command(
        self, command: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        if command == "prefetch":
            if request.get("quiesced") is not True:
                raise RuntimeError("prefetch requires quiesced=true")
            layer = int(request["layer"])
            experts = [int(item) for item in request.get("experts", [])]
            mapping = self.data_plane.prefetch(layer, experts)
            return {"mapping": mapping, "status": self.data_plane.layer_status(layer)}
        if command == "evict":
            if request.get("quiesced") is not True:
                raise RuntimeError("evict requires quiesced=true")
            layer = int(request["layer"])
            experts = [int(item) for item in request.get("experts", [])]
            return {
                "evicted": self.data_plane.evict(layer, experts),
                "status": self.data_plane.layer_status(layer),
            }
        if command == "resize":
            if request.get("quiesced") is not True:
                raise RuntimeError("resize requires a token-boundary quiesced=true guard")
            requested_by_layer = {
                int(layer): int(slots)
                for layer, slots in dict(request.get("slots_by_layer") or {}).items()
            }
            uniform_slots = request.get("slots_per_layer")
            if uniform_slots is None and not requested_by_layer:
                raise ValueError("resize requires slots_per_layer or slots_by_layer")
            retain_policy = str(request.get("retain_policy", "lru"))
            if request.get("miss_debt_budget_ms") is not None:
                budget = float(request["miss_debt_budget_ms"])
                if budget <= 0:
                    raise ValueError("miss debt budget must be positive")
                self._miss_debt_budget_ms = budget
            self._reset_miss_debt()
            state_before = self.state_island_status(sample_content=True)
            results = []
            for layer in self.data_plane.layers():
                before = self.data_plane.layer_status(layer, include_mapping=False)
                current_slots = int(before["slot_count"])
                global_experts = int(before["global_experts"])
                slots = int(
                    requested_by_layer.get(
                        layer,
                        uniform_slots if uniform_slots is not None else current_slots,
                    )
                )
                if slots < 22 or slots > global_experts:
                    raise ValueError(
                        f"layer {layer} slot target must be within [22, {global_experts}]"
                    )
                if slots == current_slots:
                    results.append(before)
                    continue
                retain = (
                    self.route_window.hot_experts(layer, slots)
                    if retain_policy == "decode_hot"
                    else None
                )
                preserve_retained = slots < current_slots
                if slots > current_slots:
                    retain = (
                        list(range(global_experts))
                        if slots >= global_experts
                        else self.route_window.hot_experts(layer, slots)
                    )
                result = self.data_plane.resize(
                    layer,
                    slots,
                    retain=retain,
                    preserve_retained=preserve_retained,
                )
                results.append(result)
            state_after = self.state_island_status(sample_content=True)
            checked = bool(state_before.get("attached"))
            preserved = (
                state_before.get("allocation_fingerprint")
                == state_after.get("allocation_fingerprint")
                and state_before.get("allocated_bytes")
                == state_after.get("allocated_bytes")
                and state_before.get("content_sample_fingerprint")
                == state_after.get("content_sample_fingerprint")
                if checked
                else None
            )
            self._last_state_island_guard = {
                "checked": checked,
                "preserved": preserved,
                "before_bytes": int(state_before.get("allocated_bytes", 0)),
                "after_bytes": int(state_after.get("allocated_bytes", 0)),
                "copy_bytes": int(state_before.get("copy_bytes", 0))
                + int(state_after.get("copy_bytes", 0)),
                "content_sampled": checked,
            }
            if checked and not preserved:
                raise RuntimeError(
                    "expert resize changed the live KV/Mamba state island allocation"
                )
            return {
                "resized_layers": len(results),
                "slots_per_layer": (
                    int(uniform_slots) if uniform_slots is not None else None
                ),
                "slots_by_layer": {
                    str(item["layer"]): int(item["slot_count"]) for item in results
                },
                "retain_policy": retain_policy,
                "state_island_guard": dict(self._last_state_island_guard),
            }
        if command == "phase":
            phase = str(request.get("phase", "idle"))
            self.route_window.set_phase(
                phase, reset_decode=bool(request.get("reset_decode", False))
            )
            if bool(request.get("reset_decode", False)):
                self._reset_miss_debt()
            return self.route_window.status()
        if command == "suspend":
            if request.get("quiesced") is not True:
                raise RuntimeError("suspend requires quiesced=true")
            evicted = self.data_plane.evict_all()
            self.suspended = True
            return {"suspended": True, "evicted": evicted}
        if command == "resume":
            self.suspended = False
            return {"suspended": False, "layers": len(self.data_plane.layers())}
        if command == "evict_all":
            if request.get("quiesced") is not True:
                raise RuntimeError("evict_all requires quiesced=true")
            return {"evicted": self.data_plane.evict_all()}
        raise ValueError(f"unknown EER command: {command}")

    def status(self) -> dict[str, Any]:
        data_plane = self.data_plane.status()
        layers_ready = len(data_plane.get("layers", [])) == self.expected_layers
        data_plane["data_plane_ready"] = bool(
            data_plane.get("data_plane_ready")
            and layers_ready
            and not self.suspended
            and not self.transitioning
            and not self.faulted
        )
        live_slots = {
            int(item["slot_count"]) for item in data_plane.get("layers", [])
        }
        slots_by_layer = {
            str(item["layer"]): int(item["slot_count"])
            for item in data_plane.get("layers", [])
        }
        return {
            "mode": self.config.mode,
            "backend": "vllm_modelopt_nvfp4_marlin",
            "data_plane_ready": (
                self.config.mode == "elastic" and data_plane["data_plane_ready"]
            ),
            "exact_route_required": True,
            "suspended": self.suspended,
            "transitioning": self.transitioning,
            "faulted": self.faulted,
            "model_fingerprint": self.fingerprint,
            "slots_per_layer": (
                next(iter(live_slots))
                if len(live_slots) == 1
                else min(live_slots, default=self.config.slots_per_layer)
            ),
            "slots_by_layer": slots_by_layer,
            "minimum_slots_per_layer": min(
                live_slots, default=self.config.slots_per_layer
            ),
            "maximum_slots_per_layer": max(
                live_slots, default=self.config.slots_per_layer
            ),
            "cache": self.local_store.status(),
            "rdma": {
                "enabled": bool(self.config.rdma_peer),
                "peer": self.config.rdma_peer,
                "port": (
                    self.config.rdma_pool_port
                    if self.remote_pool is not None
                    else self.config.rdma_port
                ),
                "path": (
                    "connectx_persistent_qp_to_pinned_staging_to_cuda_slot"
                    if self.remote_pool is not None
                    else "legacy_object_get_to_local_ssd_cache_to_cuda_slot"
                ),
                "pool": (
                    self.remote_pool.status()
                    if self.remote_pool is not None
                    else None
                ),
                "gpudirect_claimed": False,
            },
            "exported_objects": self.exported_objects,
            "expected_objects": self.expected_objects,
            "expected_layers": self.expected_layers,
            "export_complete": self._export_manifest_valid(),
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "last_error": self.last_error,
            "data_plane": data_plane,
            "route_trace": self.route_window.status(),
            "miss_debt": self._miss_debt_status(),
            "state_island": self.state_island_status(),
        }

    def _reset_miss_debt(self) -> None:
        with self._miss_debt_lock:
            self._miss_debt_ms = 0.0
            self._miss_debt_load_ms = 0.0
            self._miss_debt_tokens = 0
            self._miss_debt_violations = 0
            self._miss_debt_exceeded = False
            self._current_token_load_ms = 0.0

    def _miss_debt_status(self) -> dict[str, Any]:
        with self._miss_debt_lock:
            return {
                "budget_ms_per_token": self._miss_debt_budget_ms,
                "debt_ms": self._miss_debt_ms,
                "blocking_load_ms": self._miss_debt_load_ms,
                "decode_tokens": self._miss_debt_tokens,
                "violations": self._miss_debt_violations,
                "exceeded": self._miss_debt_exceeded,
                "action": "yield_or_hibernate",
                "evidence": "runtime_remote_parse_h2d_mapping_wall",
            }

    def _write_export_manifest(self) -> None:
        self.local_store.enforce_quota()
        stored_objects = int(self.local_store.status()["objects"])
        if stored_objects < self.expected_objects:
            raise RuntimeError(
                "runtime expert cache quota is too small for a complete export: "
                f"{stored_objects}/{self.expected_objects} objects retained"
            )
        manifest = {
            "schema_version": 1,
            "complete": True,
            "format": RUNTIME_FORMAT,
            "model_fingerprint": self.fingerprint,
            "vllm_version": SUPPORTED_VLLM,
            "objects": stored_objects,
            "created_at": time.time(),
        }
        destination = self.config.cache_dir / "runtime-manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(
            self.config.cache_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _fetch_remote_manifest(self, remote_store: RDMAExpertStore) -> None:
        destination = self.config.cache_dir / "runtime-manifest.json"
        temporary = self.config.cache_dir / (
            f".runtime-manifest.{os.getpid()}.remote"
        )
        temporary.unlink(missing_ok=True)
        try:
            remote_store.transport.get("runtime-manifest.json", temporary)
            payload = json.loads(temporary.read_text(encoding="utf-8"))
            if not self._manifest_payload_valid(payload):
                raise ValueError("remote runtime manifest is incomplete or incompatible")
            os.replace(temporary, destination)
            descriptor = os.open(
                self.config.cache_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def _export_manifest_valid(self) -> bool:
        path = self.config.cache_dir / "runtime-manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return self._manifest_payload_valid(payload)

    def _manifest_payload_valid(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        return bool(
            payload.get("schema_version") == 1
            and payload.get("complete")
            and payload.get("format") == RUNTIME_FORMAT
            and payload.get("model_fingerprint") == self.fingerprint
            and payload.get("vllm_version") == SUPPORTED_VLLM
            and int(payload.get("objects", 0)) >= self.expected_objects
        )


class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    runtime: EERRuntime


class _ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline(1024 * 1024)
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            response = {"ok": True, "result": self.server.runtime.handle_command(request)}
        except Exception as exc:
            self.server.runtime.last_error = str(exc)
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")


_RUNTIME: EERRuntime | None = None
_INSTALLED = False


def install() -> EERRuntime | None:
    global _INSTALLED, _RUNTIME
    if _INSTALLED:
        return _RUNTIME
    config = EERRuntimeConfig.from_environment()
    if config.mode == "off":
        _INSTALLED = True
        return None
    version = importlib.metadata.version("vllm")
    if version != SUPPORTED_VLLM:
        raise RuntimeError(f"PLLM EER requires vLLM {SUPPORTED_VLLM}, found {version}")
    if config.mode == "elastic" and config.slots_per_layer < 22:
        raise ValueError("PLLM EER slots_per_layer cannot be below Nemotron Top-22")

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
    from vllm.model_executor.model_loader import ep_weight_filter
    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runtime = EERRuntime(config)
    original_create = ModelOptNvFp4FusedMoE.create_weights
    original_routed_init = RoutedExperts.__init__
    original_map = RoutedExperts._map_global_expert_id_to_local_expert_id
    original_process = ModelOptNvFp4FusedMoE.process_weights_after_loading
    original_apply = ModelOptNvFp4FusedMoE.apply
    original_init_filter = DefaultModelLoader._init_ep_weight_filter
    original_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    original_init_fp8_kv_scales = GPUModelRunner.init_fp8_kv_scales

    def create_weights(method, layer, num_experts, *args, **kwargs):
        effective = (
            min(config.slots_per_layer, num_experts)
            if config.mode == "elastic"
            else num_experts
        )
        result = original_create(method, layer, effective, *args, **kwargs)
        if config.mode == "elastic":
            for parameter in layer.parameters(recurse=False):
                parameter.data.zero_()
        return result

    def routed_init(layer, *args, **kwargs):
        import torch

        original_routed_init(layer, *args, **kwargs)
        if config.mode != "elastic" or not isinstance(
            layer.quant_method, ModelOptNvFp4FusedMoE
        ):
            return
        slots = int(layer.num_experts)
        layer.local_num_experts = slots
        layer.moe_config.num_local_experts = slots
        layer._pllm_eer_logical_to_slot = {
            logical: logical for logical in range(slots)
        }
        mapping = layer.w13_weight.new_full(
            (layer.global_num_experts,), -1, dtype=torch.int32
        )
        mapping[:slots] = torch.arange(
            slots, dtype=torch.int32, device=mapping.device
        )
        layer._expert_map = mapping

    def map_expert(layer, expert_id):
        mapping = getattr(layer, "_pllm_eer_logical_to_slot", None)
        if mapping is not None:
            return mapping.get(expert_id, -1)
        return original_map(layer, expert_id)

    def process_weights(method, layer):
        runtime.release_loader_cache()
        result = original_process(method, layer)
        runtime.initialize_layer(layer)
        return result

    def apply(method, layer, x, topk_weights, topk_ids, *args, **kwargs):
        runtime.ensure(layer, topk_ids)
        return original_apply(
            method, layer, x, topk_weights, topk_ids, *args, **kwargs
        )

    def init_expert_filter(loader, model_config):
        original_init_filter(loader, model_config)
        if config.mode == "elastic":
            loader.local_expert_ids = set()

    def should_skip_weight(weight_name, local_expert_ids):
        if config.mode != "elastic" or local_expert_ids is None:
            return ep_weight_filter_should_skip(weight_name, local_expert_ids)
        expert_id = ep_weight_filter.parse_expert_id(weight_name)
        return expert_id is not None and expert_id not in local_expert_ids

    def init_fp8_kv_scales(model_runner):
        runtime.bind_model_runner(model_runner)
        caches = model_runner.kv_caches
        model_runner.kv_caches = flatten_cache_tensors(caches)
        try:
            return original_init_fp8_kv_scales(model_runner)
        finally:
            model_runner.kv_caches = caches

    def initialize_kv_cache(model_runner, *args, **kwargs):
        result = original_initialize_kv_cache(model_runner, *args, **kwargs)
        runtime.bind_model_runner(model_runner)
        return result

    ep_weight_filter_should_skip = ep_weight_filter.should_skip_weight

    ModelOptNvFp4FusedMoE.create_weights = create_weights
    RoutedExperts.__init__ = routed_init
    RoutedExperts._map_global_expert_id_to_local_expert_id = map_expert
    ModelOptNvFp4FusedMoE.process_weights_after_loading = process_weights
    ModelOptNvFp4FusedMoE.apply = apply
    DefaultModelLoader._init_ep_weight_filter = init_expert_filter
    ep_weight_filter.should_skip_weight = should_skip_weight
    weight_utils.should_skip_weight = should_skip_weight
    GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    GPUModelRunner.init_fp8_kv_scales = init_fp8_kv_scales
    marlin_utils_fp4._nvfp4_compute_scale_factor = (
        low_memory_nvfp4_scale_factor
    )
    _RUNTIME = runtime
    _INSTALLED = True
    return runtime


def request_runtime(
    socket_path: str | Path, request: dict[str, Any], timeout: float = 30.0
) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        chunks = bytearray()
        while not chunks.endswith(b"\n"):
            block = client.recv(65536)
            if not block:
                break
            chunks.extend(block)
    response = json.loads(chunks)
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "EER runtime command failed")))
    return response["result"]


def _layer_id(layer: Any) -> int:
    name = str(getattr(layer, "layer_name", ""))
    match = LAYER_PATTERN.search(name)
    if match is None:
        raise ValueError(f"cannot derive Nemotron layer ID from {name!r}")
    return int(match.group("layer"))


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")
