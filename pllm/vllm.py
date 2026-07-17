from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable
from urllib.parse import urlparse

import psutil
import requests

from .models import VLLMService


VLLM_COMMAND_MARKERS = (
    "vllm serve",
    "vllm.entrypoints.openai.api_server",
    "vllm.entrypoints.serve",
)


class VLLMClient:
    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()

    def probe(
        self,
        base_url: str,
        pid: int | None = None,
        related_pids: list[int] | None = None,
        command: str = "",
        managed: bool = False,
    ) -> VLLMService:
        normalized = base_url.rstrip("/")
        service = VLLMService(
            service_id=_service_id(normalized),
            base_url=normalized,
            pid=pid,
            related_pids=related_pids or ([pid] if pid else []),
            command=command,
            managed=managed,
        )
        try:
            response = self.session.get(
                f"{normalized}/v1/models", timeout=min(self.timeout, 1.5)
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            if models:
                service.model = str(models[0].get("id", ""))
            service.healthy = True
        except (requests.RequestException, ValueError) as exc:
            service.last_error = str(exc)
            return service

        try:
            response = self.session.get(
                f"{normalized}/is_sleeping", timeout=min(self.timeout, 1.5)
            )
            if response.status_code == 200:
                service.controllable = True
                service.sleeping = _parse_sleeping(response)
        except requests.RequestException as exc:
            service.last_error = str(exc)
        return service

    def sleep(self, service: VLLMService, level: int, mode: str = "keep") -> None:
        if not service.controllable:
            raise RuntimeError(f"vLLM service {service.base_url} has no Sleep API")
        if level not in (0, 1, 2):
            raise ValueError(f"Unsupported vLLM sleep level: {level}")
        if mode not in {"keep", "wait", "abort"}:
            raise ValueError(f"Unsupported vLLM pause mode: {mode}")
        if level > 0:
            self._post(service, "/sleep", params={"level": 0, "mode": mode})
        self._post(service, "/sleep", params={"level": level, "mode": mode})

    def sleep_from_quiesced(
        self, service: VLLMService, level: int, mode: str = "keep"
    ) -> None:
        if not service.controllable:
            raise RuntimeError(f"vLLM service {service.base_url} has no Sleep API")
        if level not in (1, 2):
            raise ValueError("sleep_from_quiesced only accepts Level 1 or 2")
        if mode not in {"keep", "wait", "abort"}:
            raise ValueError(f"Unsupported vLLM pause mode: {mode}")
        self._post(service, "/sleep", params={"level": level, "mode": mode})

    def wake(self, service: VLLMService, level: int | None) -> None:
        if not service.controllable:
            raise RuntimeError(f"vLLM service {service.base_url} has no Sleep API")
        if level == 0:
            self._post(service, "/wake_up", params=[("tags", "scheduling")])
            return
        if level == 2:
            self._post(service, "/wake_up", params=[("tags", "weights")])
            self._post(
                service,
                "/collective_rpc",
                json={"method": "reload_weights"},
                timeout=max(self.timeout, 120.0),
            )
            self._post(service, "/wake_up", params=[("tags", "kv_cache")])
            return
        self._post(service, "/wake_up")

    def _post(
        self,
        service: VLLMService,
        path: str,
        *,
        params=None,
        json=None,
        timeout: float | None = None,
    ) -> requests.Response:
        response = self.session.post(
            f"{service.base_url}{path}",
            params=params,
            json=json,
            timeout=timeout or max(self.timeout, 300.0),
        )
        response.raise_for_status()
        return response


class VLLMDiscovery:
    def __init__(
        self,
        client: VLLMClient,
        configured_urls: Iterable[str],
        excluded_patterns: Iterable[str] = (),
    ) -> None:
        self.client = client
        self.configured_urls = list(configured_urls)
        self.excluded_patterns = tuple(item.lower() for item in excluded_patterns)

    def discover(self) -> list[VLLMService]:
        candidates: dict[str, tuple[int | None, list[int], str, bool]] = {
            url.rstrip("/"): (None, [], "", False) for url in self.configured_urls
        }
        for pid, related_pids, command, urls in self._process_candidates():
            for url in urls:
                candidates[url] = (pid, related_pids, command, False)

        services = []
        for url, (pid, related_pids, command, managed) in sorted(candidates.items()):
            services.append(
                self.client.probe(url, pid, related_pids, command, managed)
            )
        return services

    def _process_candidates(self) -> list[tuple[int, list[int], str, list[str]]]:
        result = []
        for process in psutil.process_iter(["pid", "cmdline", "name"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
                lowered = command.lower()
                if not is_standalone_vllm_command(lowered, self.excluded_patterns):
                    continue
                urls = _listening_urls(process)
                if urls:
                    result.append(
                        (int(process.pid), _process_tree_pids(process), command, urls)
                    )
            except (psutil.Error, OSError):
                continue
        return result


class VLLMManager:
    def __init__(self, client: VLLMClient, discovery: VLLMDiscovery) -> None:
        self.client = client
        self.discovery = discovery
        self.services: list[VLLMService] = []
        self._last_levels: dict[str, int] = {}

    def refresh(self) -> list[VLLMService]:
        previous = {service.service_id: service for service in self.services}
        refreshed = []
        for service in self.discovery.discover():
            old = previous.get(service.service_id)
            if old and old.last_sleep_level is not None:
                service = replace(service, last_sleep_level=old.last_sleep_level)
            refreshed.append(service)
        self.services = refreshed
        return list(self.services)

    def controllable(self) -> list[VLLMService]:
        return [
            service
            for service in self.services
            if service.healthy and service.controllable
        ]

    def sleep_all(self, level: int, mode: str = "keep") -> int:
        controlled = self.controllable()
        if not controlled:
            raise RuntimeError("No healthy vLLM service with Sleep Mode is available")
        for service in controlled:
            self.client.sleep(service, level, mode)
            self._last_levels[service.service_id] = level
            service.sleeping = True
            service.last_sleep_level = level
            service.last_pause_mode = mode
        return len(controlled)

    def deep_sleep_all_from_quiesced(self, level: int, mode: str = "keep") -> int:
        controlled = self.controllable()
        if not controlled:
            raise RuntimeError("No healthy vLLM service with Sleep Mode is available")
        for service in controlled:
            self.client.sleep_from_quiesced(service, level, mode)
            self._last_levels[service.service_id] = level
            service.sleeping = True
            service.last_sleep_level = level
            service.last_pause_mode = mode
        return len(controlled)

    def wake_all(self) -> int:
        controlled = self.controllable()
        if not controlled:
            raise RuntimeError("No healthy vLLM service with Sleep Mode is available")
        for service in controlled:
            level = self._last_levels.get(service.service_id, service.last_sleep_level)
            self.client.wake(service, level)
            service.sleeping = False
        return len(controlled)

    def target_url(self) -> str | None:
        for service in self.services:
            if service.healthy and not service.sleeping:
                return service.base_url
        for service in self.services:
            if service.healthy:
                return service.base_url
        return None


def is_standalone_vllm_command(
    command: str, excluded_patterns: Iterable[str] = ()
) -> bool:
    lowered = command.lower()
    if any(pattern in lowered for pattern in excluded_patterns):
        return False
    return any(marker in lowered for marker in VLLM_COMMAND_MARKERS)


def _listening_urls(process: psutil.Process) -> list[str]:
    urls = []
    try:
        connections = process.net_connections(kind="inet")
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return urls
    for connection in connections:
        if connection.status != psutil.CONN_LISTEN or not connection.laddr:
            continue
        host = connection.laddr.ip
        if host in {"0.0.0.0", "::", "::0"}:
            host = "127.0.0.1"
        if host not in {"127.0.0.1", "::1", "localhost"}:
            continue
        host_literal = f"[{host}]" if ":" in host else host
        urls.append(f"http://{host_literal}:{connection.laddr.port}")
    return sorted(set(urls))


def _process_tree_pids(process: psutil.Process) -> list[int]:
    pids = {int(process.pid)}
    try:
        pids.update(int(child.pid) for child in process.children(recursive=True))
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        pass
    return sorted(pids)


def _parse_sleeping(response: requests.Response) -> bool:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip().lower() in {"1", "true", "yes"}
    if isinstance(data, bool):
        return data
    if isinstance(data, dict):
        return bool(data.get("is_sleeping", data.get("sleeping", False)))
    return False


def _service_id(base_url: str) -> str:
    parsed = urlparse(base_url)
    label = parsed.netloc or base_url
    digest = hashlib.sha1(base_url.encode("utf-8")).hexdigest()[:8]
    return f"{label}-{digest}"
