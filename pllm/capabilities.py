from __future__ import annotations

import ctypes
import ctypes.util
import importlib.metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from .config import PLLMConfig
from .hibercache import HiberCacheManager


class CapabilityProbe:
    def __init__(self, config: PLLMConfig, hibercache: HiberCacheManager) -> None:
        self.config = config
        self.hibercache = hibercache
        self._cached: dict[str, Any] = {}
        self._last_probe = 0.0

    def collect(self, refresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if self._cached and not refresh and now - self._last_probe < 30.0:
            result = dict(self._cached)
            result["hibercache"] = self.hibercache.status()
            return result
        gpu_name = self._gpu_name()
        uma = "gb10" in gpu_name.lower() or "dgx spark" in gpu_name.lower()
        gdr = self._cuda_attribute(116)
        dma_buf = self._cuda_attribute(124)
        hcas = self._rdma_devices()
        model = Path(self.config.model_path)
        model_metadata = self._model_metadata(model)
        model_bytes = sum(
            (path.stat().st_size for path in model.glob("*.safetensors")), start=0
        ) if model.exists() else 0
        disk = shutil.disk_usage(model if model.exists() else Path("/"))
        fastsafetensors_version = self._version("fastsafetensors")
        unified_loader = self._fastsafetensors_has_unified_loader()
        patch = self._hibercache_patch_status()
        continuity = self._continuity_validation()
        self._cached = {
            "platform": {
                "machine": platform.machine(),
                "system": platform.system(),
                "gpu_name": gpu_name,
                "dgx_spark": uma,
                "uma": uma,
            },
            "cuda": {
                "gpudirect_rdma": gdr,
                "dma_buf": dma_buf,
                "rdma_path": (
                    "gpu_direct" if gdr is True and not uma else "host_staging"
                ),
            },
            "vllm": {
                "version": self._version("vllm"),
                "sleep_mode": True,
                "keep_pause": True,
                "hibercache_patch": patch,
            },
            "sparkload": {
                "selected": self._loader_choice(uma, fastsafetensors_version),
                "fastsafetensors_version": fastsafetensors_version,
                "unified_loader": unified_loader,
                "model_path": str(model),
                "model_size_gb": round(model_bytes / 1024**3, 2),
                "architecture": model_metadata["architecture"],
                "quant_method": model_metadata["quant_method"],
                "model_read_only": not os.access(model, os.W_OK) if model.exists() else None,
                "disk_free_gb": round(disk.free / 1024**3, 1),
            },
            "rdma": {
                "available": bool(hcas),
                "peer": self.config.rdma_peer,
                "devices": hcas,
                "model_express": bool(self._version("modelexpress")),
                "nixl": bool(self._version("nixl")),
            },
            "hibercache": self.hibercache.status(refresh=True),
            "continuity": {
                "mock_same_stream_validated": True,
                "real_model_validated": continuity["validated"],
                "validation_result": continuity["path"],
                "hybrid_state_strategy": "connector_attention_plus_token_recompute",
            },
        }
        self._last_probe = now
        return dict(self._cached)

    def _loader_choice(self, uma: bool, fastsafetensors_version: str) -> str:
        if self.config.loader_mode != "auto":
            return self.config.loader_mode
        if fastsafetensors_version:
            return "fastsafetensors_unified" if uma else "fastsafetensors_gds"
        return "multithread_safetensors"

    @staticmethod
    def _version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return ""

    @staticmethod
    def _gpu_name() -> str:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                value = pynvml.nvmlDeviceGetName(handle)
                return value.decode() if isinstance(value, bytes) else str(value)
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            return ""

    @staticmethod
    def _cuda_attribute(attribute: int) -> bool | None:
        library = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            cuda = ctypes.CDLL(library)
            if cuda.cuInit(0) != 0:
                return None
            device = ctypes.c_int()
            if cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
                return None
            value = ctypes.c_int()
            result = cuda.cuDeviceGetAttribute(
                ctypes.byref(value), attribute, device.value
            )
            return bool(value.value) if result == 0 else None
        except (OSError, AttributeError):
            return None

    @staticmethod
    def _rdma_devices() -> list[dict[str, Any]]:
        root = Path("/sys/class/infiniband")
        result = []
        if not root.exists():
            return result
        for device in sorted(root.iterdir()):
            port = device / "ports" / "1"
            result.append(
                {
                    "name": device.name,
                    "state": _read_text(port / "state"),
                    "physical_state": _read_text(port / "phys_state"),
                    "rate": _read_text(port / "rate"),
                }
            )
        return result

    @staticmethod
    def _fastsafetensors_has_unified_loader() -> bool:
        try:
            dist = importlib.metadata.distribution("fastsafetensors")
            path = Path(dist.locate_file("fastsafetensors/copier/unified.py"))
            return path.exists() and "DGX Spark" in path.read_text(
                encoding="utf-8", errors="ignore"
            )
        except (importlib.metadata.PackageNotFoundError, OSError):
            return False

    @staticmethod
    def _hibercache_patch_status() -> dict[str, Any]:
        try:
            dist = importlib.metadata.distribution("vllm")
            path = Path(dist.locate_file("vllm/v1/engine/core.py"))
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {
                "required": True,
                "installed": "PLLM_HIBERCACHE_PRESERVE_CONNECTOR" in text,
                "target": str(path),
            }
        except (importlib.metadata.PackageNotFoundError, OSError):
            return {"required": True, "installed": False, "target": ""}

    @staticmethod
    def _continuity_validation() -> dict[str, Any]:
        path = Path(__file__).resolve().parent.parent / "results" / "continuity_level2.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validated = bool(
                payload.get("exact_match")
                and payload.get("stream_finished")
                and payload.get("chunks_during_pause") == 0
            )
            return {"validated": validated, "path": str(path)}
        except (OSError, ValueError, TypeError):
            return {"validated": False, "path": ""}

    @staticmethod
    def _model_metadata(model: Path) -> dict[str, str]:
        try:
            payload = json.loads((model / "config.json").read_text(encoding="utf-8"))
            architectures = payload.get("architectures") or []
            quantization = payload.get("quantization_config") or {}
            return {
                "architecture": str(architectures[0]) if architectures else "",
                "quant_method": str(quantization.get("quant_method", "")),
            }
        except (OSError, ValueError, TypeError):
            return {"architecture": "", "quant_method": ""}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
