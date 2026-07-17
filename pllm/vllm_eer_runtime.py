from __future__ import annotations

import gc
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

from .expert_dataplane import ExpertSlotDataPlane
from .expert_catalog import ExpertCatalog
from .expert_store import (
    ExpertPayload,
    RDMAExpertStore,
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

    @classmethod
    def from_environment(cls) -> "EERRuntimeConfig":
        mode = os.getenv("PLLM_EER_MODE", "off").strip().lower()
        if mode not in {"off", "export", "elastic"}:
            raise ValueError("PLLM_EER_MODE must be off, export, or elastic")
        runtime_dir = Path(
            os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        )
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
                os.getenv("PLLM_EER_CACHE_DIR", "/mnt/ssd-storage/pllm-experts")
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
            content = tensor.detach().contiguous().view(self.torch.uint8).cpu()
            tensors.append(
                (
                    name,
                    _dtype_name(dtype),
                    tuple(shape),
                    content.numpy().tobytes(),
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
            cpu_bytes = self.torch.frombuffer(bytearray(raw), dtype=self.torch.uint8)
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
        self.local_store = SSDExpertStore(
            config.cache_dir,
            model_fingerprint=self.fingerprint,
            quota_bytes=config.cache_quota_bytes,
            required_format=RUNTIME_FORMAT,
        )
        sources: list[Any] = [self.local_store]
        remote_store = None
        if config.rdma_peer:
            if config.rdma_token_file is None or not config.rdma_token_file.is_file():
                raise RuntimeError("RDMA warm source requires PLLM_EER_RDMA_TOKEN_FILE")
            remote_store = RDMAExpertStore(
                peer=config.rdma_peer,
                port=config.rdma_port,
                binary=config.rdma_binary,
                local_cache=self.local_store,
                token_file=config.rdma_token_file or "",
                allocator=config.rdma_allocator,
                device=config.rdma_device,
            )
            sources.append(remote_store)
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
        self.data_plane = ExpertSlotDataPlane(TieredExpertSource(sources))
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
            backend = str(layer.quant_method.nvfp4_backend.value)
            if backend != "marlin":
                raise RuntimeError(f"PLLM EER requires the Marlin MoE backend, found {backend}")
            if layer.quant_method.is_monolithic:
                raise RuntimeError("PLLM EER requires a modular MoE kernel with visible Top-k IDs")
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

    def ensure(self, layer: Any, topk_ids: Any) -> None:
        if self.config.mode != "elastic":
            return
        if self.suspended:
            raise RuntimeError("PLLM EER data plane is suspended")
        layer_id = _layer_id(layer)
        experts = [
            int(item)
            for item in topk_ids.detach().unique().cpu().tolist()
            if int(item) >= 0
        ]
        self.data_plane.ensure(layer_id, experts, reason="actual_topk")

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
            slots = int(request["slots_per_layer"])
            results = []
            for layer in self.data_plane.layers():
                results.append(self.data_plane.resize(layer, slots))
            return {"resized_layers": len(results), "slots_per_layer": slots}
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
                else self.config.slots_per_layer
            ),
            "cache": self.local_store.status(),
            "rdma": {
                "enabled": bool(self.config.rdma_peer),
                "peer": self.config.rdma_peer,
                "port": self.config.rdma_port,
                "path": (
                    "connectx_to_registered_host_buffer_to_local_ssd_cache_"
                    "to_cuda_slot"
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
    from vllm.model_executor.model_loader import ep_weight_filter
    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    runtime = EERRuntime(config)
    original_create = ModelOptNvFp4FusedMoE.create_weights
    original_routed_init = RoutedExperts.__init__
    original_map = RoutedExperts._map_global_expert_id_to_local_expert_id
    original_process = ModelOptNvFp4FusedMoE.process_weights_after_loading
    original_apply = ModelOptNvFp4FusedMoE.apply
    original_init_filter = DefaultModelLoader._init_ep_weight_filter

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

    ep_weight_filter_should_skip = ep_weight_filter.should_skip_weight

    ModelOptNvFp4FusedMoE.create_weights = create_weights
    RoutedExperts.__init__ = routed_init
    RoutedExperts._map_global_expert_id_to_local_expert_id = map_expert
    ModelOptNvFp4FusedMoE.process_weights_after_loading = process_weights
    ModelOptNvFp4FusedMoE.apply = apply
    DefaultModelLoader._init_ep_weight_filter = init_expert_filter
    ep_weight_filter.should_skip_weight = should_skip_weight
    weight_utils.should_skip_weight = should_skip_weight
    runtime.start_control_server()
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
