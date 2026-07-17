from __future__ import annotations

import os
import platform
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import psutil

from .foreground import ForegroundProvider
from .models import ProcessGpuUsage, SensorSnapshot

try:
    import pynvml
except ImportError:  # pragma: no cover - exercised on systems without NVML
    pynvml = None


GIB = 1024**3


class SystemMonitor:
    def __init__(self, foreground: ForegroundProvider | None = None) -> None:
        self.foreground = foreground or ForegroundProvider()
        self._nvml_ready = False
        self._handle = None
        self._last_process_timestamp = 0
        self._last_process_refresh = 0.0
        self._cached_processes: list[ProcessGpuUsage] = []
        self._gpu_name = ""
        self._gpu_memory_total_gb: float | None = None
        self._gpu_power_limit_watts: float | None = None
        self._gpu_uma = False
        self._last_power_refresh = 0.0
        self._last_system_refresh = 0.0
        self._system_cache: dict[str, float] = {}
        self._last_gpu_slow_refresh = 0.0
        self._gpu_slow_cache: dict[str, float | int | None] = {
            "power_watts": None,
            "temperature_c": None,
        }
        self._power_profile = "unknown"
        self._battery = (False, None)
        self._initialize_nvml()

    def _initialize_nvml(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ready = True
            raw_name = pynvml.nvmlDeviceGetName(self._handle)
            self._gpu_name = (
                raw_name.decode(errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            self._gpu_uma = _looks_like_dgx_spark(self._gpu_name)
            try:
                self._gpu_memory_total_gb = (
                    pynvml.nvmlDeviceGetMemoryInfo(self._handle).total / GIB
                )
            except Exception:
                self._gpu_uma = True
            self._gpu_power_limit_watts = _nvml_value(
                lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000
            )
        except Exception:
            self._nvml_ready = False
            self._handle = None

    def close(self) -> None:
        if self._nvml_ready and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def collect(self) -> SensorSnapshot:
        now = time.time()
        monotonic_now = time.monotonic()
        if monotonic_now - self._last_system_refresh >= 1.0:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            psi_some, psi_full = _read_memory_psi()
            self._system_cache = {
                "memory_total_gb": memory.total / GIB,
                "memory_available_gb": memory.available / GIB,
                "swap_used_gb": swap.used / GIB,
                "memory_psi_some": psi_some,
                "memory_psi_full": psi_full,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "load_average": os.getloadavg()[0],
            }
            self._last_system_refresh = monotonic_now
        if time.monotonic() - self._last_power_refresh >= 30.0:
            self._refresh_power_state()

        snapshot = SensorSnapshot(
            timestamp=now,
            memory_total_gb=self._system_cache.get("memory_total_gb", 0.0),
            memory_available_gb=self._system_cache.get("memory_available_gb", 0.0),
            swap_used_gb=self._system_cache.get("swap_used_gb", 0.0),
            memory_psi_some=self._system_cache.get("memory_psi_some", 0.0),
            memory_psi_full=self._system_cache.get("memory_psi_full", 0.0),
            cpu_percent=self._system_cache.get("cpu_percent", 0.0),
            load_average=self._system_cache.get("load_average", 0.0),
            on_battery=self._battery[0],
            battery_percent=self._battery[1],
            power_profile=self._power_profile,
            foreground=self.foreground.get(),
        )
        self._populate_gpu(snapshot)
        return snapshot

    def _populate_gpu(self, snapshot: SensorSnapshot) -> None:
        if not self._nvml_ready or self._handle is None or pynvml is None:
            snapshot.uma = _looks_like_dgx_spark("")
            return
        try:
            snapshot.gpu_available = True
            snapshot.gpu_name = self._gpu_name
            snapshot.gpu_util = int(
                pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            )
            snapshot.uma = self._gpu_uma
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                snapshot.gpu_memory_total_gb = self._gpu_memory_total_gb
                snapshot.gpu_memory_used_gb = info.used / GIB
                snapshot.gpu_memory_free_gb = info.free / GIB
            except Exception:
                snapshot.uma = True
            monotonic_now = time.monotonic()
            if monotonic_now - self._last_gpu_slow_refresh >= 1.0:
                self._gpu_slow_cache = {
                    "power_watts": _nvml_value(
                        lambda: pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000
                    ),
                    "temperature_c": _nvml_value(
                        lambda: int(
                            pynvml.nvmlDeviceGetTemperature(
                                self._handle, pynvml.NVML_TEMPERATURE_GPU
                            )
                        )
                    ),
                }
                self._last_gpu_slow_refresh = monotonic_now
            snapshot.power_watts = self._gpu_slow_cache["power_watts"]
            snapshot.power_limit_watts = self._gpu_power_limit_watts
            snapshot.temperature_c = self._gpu_slow_cache["temperature_c"]
            if monotonic_now - self._last_process_refresh >= 1.0:
                self._cached_processes = self._collect_gpu_processes()
                self._last_process_refresh = monotonic_now
            snapshot.processes = self._cached_processes
        except Exception:
            self._nvml_ready = False
            snapshot.gpu_available = False

    def _collect_gpu_processes(self) -> list[ProcessGpuUsage]:
        assert pynvml is not None and self._handle is not None
        by_pid: dict[int, ProcessGpuUsage] = {}
        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            getter = getattr(pynvml, getter_name, None)
            if getter is None:
                continue
            try:
                for process in getter(self._handle):
                    used = getattr(process, "usedGpuMemory", 0)
                    if not isinstance(used, int) or used < 0 or used > 2**63:
                        used = 0
                    by_pid.setdefault(
                        int(process.pid),
                        ProcessGpuUsage(
                            pid=int(process.pid),
                            name=_process_name(int(process.pid)),
                            memory_gb=used / GIB,
                        ),
                    )
            except Exception:
                continue

        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(
                self._handle, self._last_process_timestamp
            )
            for sample in samples:
                self._last_process_timestamp = max(
                    self._last_process_timestamp, int(sample.timeStamp)
                )
                usage = by_pid.setdefault(
                    int(sample.pid),
                    ProcessGpuUsage(
                        pid=int(sample.pid), name=_process_name(int(sample.pid))
                    ),
                )
                usage.sm_util = int(sample.smUtil)
                usage.memory_util = int(sample.memUtil)
                usage.encoder_util = int(sample.encUtil)
                usage.decoder_util = int(sample.decUtil)
        except Exception:
            pass
        return sorted(by_pid.values(), key=lambda item: item.memory_gb, reverse=True)

    def _refresh_power_state(self) -> None:
        self._last_power_refresh = time.monotonic()
        try:
            result = subprocess.run(
                ["powerprofilesctl", "get"],
                capture_output=True,
                text=True,
                timeout=0.5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._power_profile = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            self._power_profile = "unknown"
        self._battery = _read_battery()


@lru_cache(maxsize=512)
def _process_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except (psutil.Error, OSError):
        return ""


def _nvml_value(callback):
    try:
        return callback()
    except Exception:
        return None


def _read_memory_psi() -> tuple[float, float]:
    try:
        lines = Path("/proc/pressure/memory").read_text(encoding="ascii").splitlines()
    except OSError:
        return 0.0, 0.0
    values: dict[str, float] = {}
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        avg10 = next((field for field in fields if field.startswith("avg10=")), None)
        if avg10:
            values[fields[0]] = float(avg10.split("=", 1)[1])
    return values.get("some", 0.0), values.get("full", 0.0)


def _read_battery() -> tuple[bool, float | None]:
    root = Path("/sys/class/power_supply")
    if not root.exists():
        return False, None
    batteries = []
    ac_online = None
    for supply in root.iterdir():
        try:
            supply_type = (supply / "type").read_text().strip().lower()
            if supply_type == "battery":
                capacity = float((supply / "capacity").read_text().strip())
                batteries.append(capacity)
            elif supply_type in {"mains", "usb", "usb_c"}:
                online_path = supply / "online"
                if online_path.exists():
                    ac_online = online_path.read_text().strip() == "1"
        except (OSError, ValueError):
            continue
    if not batteries:
        return False, None
    return ac_online is False, sum(batteries) / len(batteries)


def _looks_like_dgx_spark(gpu_name: str) -> bool:
    name = gpu_name.lower()
    return "gb10" in name or "dgx spark" in name or (
        platform.machine() in {"aarch64", "arm64"} and "nvidia" in name
    )
