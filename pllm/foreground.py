from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .models import ForegroundApp


class ForegroundProvider:
    """Read the active GNOME window without requiring compositor privileges."""

    def __init__(self, fallback_file: str = "") -> None:
        env_file = os.getenv("PLLM_FOREGROUND_FILE", fallback_file)
        self.fallback_file = Path(env_file).expanduser() if env_file else None
        self._bus = None
        self._interface = None
        self._last_connect_attempt = 0.0

    def get(self) -> ForegroundApp:
        file_value = self._from_file()
        if file_value is not None:
            return file_value
        return self._from_dbus()

    def _from_file(self) -> ForegroundApp | None:
        if self.fallback_file is None or not self.fallback_file.exists():
            return None
        try:
            data = json.loads(self.fallback_file.read_text(encoding="utf-8"))
            return ForegroundApp(
                pid=int(data.get("pid", 0)),
                app_id=str(data.get("app_id", "")),
                title=str(data.get("title", "")),
                wm_class=str(data.get("wm_class", "")),
                available=True,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _from_dbus(self) -> ForegroundApp:
        if self._interface is None and not self._connect_dbus():
            return ForegroundApp()
        try:
            pid, app_id, title, wm_class = self._interface.call_get_active()
            return ForegroundApp(
                pid=int(pid),
                app_id=str(app_id),
                title=str(title),
                wm_class=str(wm_class),
                available=True,
            )
        except Exception:
            self._interface = None
            self._bus = None
            return ForegroundApp()

    def _connect_dbus(self) -> bool:
        now = time.monotonic()
        if now - self._last_connect_attempt < 30.0:
            return False
        self._last_connect_attempt = now
        try:
            from dbus_next import BusType
            from dbus_next.sync import MessageBus

            self._bus = MessageBus(bus_type=BusType.SESSION).connect()
            introspection = self._bus.introspect(
                "org.pllm.Foreground", "/org/pllm/Foreground"
            )
            proxy = self._bus.get_proxy_object(
                "org.pllm.Foreground", "/org/pllm/Foreground", introspection
            )
            self._interface = proxy.get_interface("org.pllm.Foreground")
            return True
        except Exception:
            self._bus = None
            self._interface = None
            return False
