from __future__ import annotations

import os
import pwd
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import PolicyMode


DEFAULT_MODEL_PATH = Path(
    "/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
)
DEFAULT_EXPERT_CACHE_DIR = (
    Path("/mnt/ssd-storage")
    / pwd.getpwuid(os.getuid()).pw_name
    / "pllm-experts"
)


def pllm_runtime_dir() -> Path:
    configured = os.getenv("XDG_RUNTIME_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(f"/run/user/{os.getuid()}"))
    for candidate in candidates:
        if candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK):
            return candidate
    return Path("/tmp") / f"pllm-{os.getuid()}"


@dataclass(slots=True)
class PLLMConfig:
    api_host: str = "127.0.0.1"
    api_port: int = 17860
    poll_interval_seconds: float = 0.25
    service_refresh_seconds: float = 30.0
    request_timeout_seconds: float = 5.0
    mode: str = PolicyMode.AUTO.value
    default_vllm_urls: list[str] = field(
        default_factory=lambda: ["http://127.0.0.1:8000"]
    )
    model_path: str = str(DEFAULT_MODEL_PATH)
    creative_hold_seconds: float = 0.5
    yield_resume_idle_seconds: float = 3.0
    resume_idle_seconds: float = 30.0
    wake_cooldown_seconds: float = 60.0
    external_gpu_pressure_percent: int = 70
    soft_gpu_pressure_percent: int = 35
    creative_gpu_percent: int = 15
    idle_gpu_percent: int = 10
    min_free_vram_gb: float = 8.0
    min_available_memory_gb: float = 20.0
    hot_sleep_memory_reserve_gb: float = 16.0
    low_battery_percent: float = 40.0
    foreground_duration_seconds: float = 300.0
    calibration_path: str = ""
    hibercache_enabled: bool = True
    hibercache_dir: str = "/mnt/ssd-storage/pllm-cache"
    hibercache_staging_mb: int = 512
    hibercache_quota_gb: float = 20.0
    preserve_connector_patch: bool = True
    hiberstate_chunk_mb: int = 64
    hiberstate_rdma_enabled: bool = False
    hiberstate_rdma_peer: str = ""
    hiberstate_rdma_port: int = 17901
    hiberstate_rdma_binary: str = "rdma_bridge/build/pllm-rdma-store"
    hiberstate_rdma_token_file: str = "~/.config/pllm/rdma-token"
    hiberstate_rdma_allocator: str = "cuda-host"
    hiberstate_rdma_device: str = ""
    hiberstate_rdma_ib_port: int = 1
    hiberstate_rdma_gid_index: int = 0
    expert_residency_enabled: bool = True
    expert_data_plane_enabled: bool = True
    # Physical layer rebuild remains opt-in until model-specific validation passes.
    expert_auto_resize_enabled: bool = False
    expert_resize_cooldown_seconds: float = 60.0
    expert_slots_per_layer: int = 128
    expert_runtime_cache_dir: str = str(DEFAULT_EXPERT_CACHE_DIR)
    expert_runtime_cache_quota_gib: float = 80.0
    expert_runtime_socket: str = ""
    expert_rdma_port: int = 17900
    expert_rdma_binary: str = "rdma_bridge/build/pllm-rdma-store"
    expert_rdma_pool_port: int = 17902
    expert_rdma_pool_binary: str = "rdma_bridge/build/pllm-rdma-pool"
    expert_rdma_pool_index: str = ""
    expert_rdma_token_file: str = "~/.config/pllm/rdma-token"
    expert_system_reserve_gib: float = 16.0
    expert_io_budget_gib_s: float = 2.0
    expert_requested_token_rate: float = 5.0
    expert_minimum_token_rate: float = 0.5
    expert_release_deadline_ms: float = 500.0
    expert_assumed_byte_hit_rate: float = 0.95
    expert_assumed_false_prefetch_ratio: float = 0.05
    decode_elastic_enabled: bool = True
    decode_planner_async: bool = True
    decode_route_window_steps: int = 256
    decode_horizon_bucket_tokens: int = 128
    decode_min_route_observations: int = 320
    decode_candidate_slots: list[int] = field(
        default_factory=lambda: [256, 320, 384, 448, 480, 496, 504]
    )
    decode_min_byte_hit_rate: float = 0.95
    decode_max_slowdown_ratio: float = 5.0
    decode_baseline_tpot_ms: float = 100.0
    decode_miss_latency_p95_ms: float = 7.5
    decode_min_heldout_windows: int = 1
    decode_target_reclaim_gib: float = 4.0
    decode_resize_copy_gib_s: float = 100.0
    decode_expand_gib_s: float = 0.75
    decode_rebuild_ms_per_layer: float = 5.0
    decode_miss_batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 22, 32]
    )
    decode_miss_batch_p95_ms: list[float] = field(
        default_factory=lambda: [0.477, 0.784, 1.513, 2.863, 26.223, 42.710, 43.897]
    )
    loader_mode: str = "auto"
    rdma_peer: str = ""
    rdma_control_port: int = 18515
    remote_weight_source_enabled: bool = False
    policy_advisor_url: str = ""
    dry_run: bool = False
    foreground_file: str = ""
    game_patterns: list[str] = field(
        default_factory=lambda: [
            "steam_app_",
            "proton",
            "wine",
            "gamescope",
            "lutris",
            "heroic",
            "unreal",
        ]
    )
    creative_patterns: list[str] = field(
        default_factory=lambda: [
            "blender",
            "davinci",
            "resolve",
            "kdenlive",
            "obs",
            "ffmpeg",
            "premiere",
            "after effects",
            "comfyui",
        ]
    )
    excluded_process_patterns: list[str] = field(
        default_factory=lambda: [
            "train",
            "trainer",
            "grpo",
            "deepspeed",
            "torchrun",
            "accelerate launch",
        ]
    )

    @classmethod
    def default_path(cls) -> Path:
        override = os.getenv("PLLM_CONFIG")
        if override:
            return Path(override).expanduser()
        return Path.home() / ".config" / "pllm" / "config.toml"

    @classmethod
    def load(cls, path: Path | None = None) -> "PLLMConfig":
        config_path = path or cls.default_path()
        if not config_path.exists():
            config = cls()
            config.save(config_path)
            return config

        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        values = raw.get("pllm", raw)
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if key not in self.__dataclass_fields__:
                continue
            if key == "mode":
                value = PolicyMode(value).value
            if key in {
                "hibercache_staging_mb",
                "rdma_control_port",
                "expert_slots_per_layer",
                "expert_rdma_port",
                "expert_rdma_pool_port",
                "hiberstate_chunk_mb",
                "hiberstate_rdma_port",
                "hiberstate_rdma_ib_port",
                "hiberstate_rdma_gid_index",
                "decode_route_window_steps",
                "decode_horizon_bucket_tokens",
                "decode_min_route_observations",
                "decode_min_heldout_windows",
            }:
                value = int(value)
            if key in {
                "poll_interval_seconds",
                "creative_hold_seconds",
                "yield_resume_idle_seconds",
                "resume_idle_seconds",
                "hibercache_quota_gb",
                "expert_system_reserve_gib",
                "expert_io_budget_gib_s",
                "expert_requested_token_rate",
                "expert_minimum_token_rate",
                "expert_release_deadline_ms",
                "expert_assumed_byte_hit_rate",
                "expert_assumed_false_prefetch_ratio",
                "expert_runtime_cache_quota_gib",
                "expert_resize_cooldown_seconds",
                "decode_min_byte_hit_rate",
                "decode_max_slowdown_ratio",
                "decode_baseline_tpot_ms",
                "decode_miss_latency_p95_ms",
                "decode_target_reclaim_gib",
                "decode_resize_copy_gib_s",
                "decode_expand_gib_s",
                "decode_rebuild_ms_per_layer",
            }:
                value = float(value)
            setattr(self, key, value)

    def save(self, path: Path | None = None) -> Path:
        config_path = path or self.default_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["[pllm]"]
        for key, value in asdict(self).items():
            lines.append(f"{key} = {_toml_value(value)}")
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return config_path

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_expert_runtime_socket(self) -> Path:
        if self.expert_runtime_socket:
            return Path(self.expert_runtime_socket).expanduser()
        return pllm_runtime_dir() / "pllm-eer.sock"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'
