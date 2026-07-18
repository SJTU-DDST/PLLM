from __future__ import annotations

from pllm.foreground import ForegroundProvider


class FakeInterface:
    async def call_get_active(self):
        return 4242, "org.blender.Blender.desktop", "PLLM Demo", "Blender"


class FakeProxy:
    def get_interface(self, name: str):
        assert name == "org.pllm.Foreground"
        return FakeInterface()


class FakeBus:
    async def introspect(self, bus_name: str, object_path: str):
        assert bus_name == "org.pllm.Foreground"
        assert object_path == "/org/pllm/Foreground"
        return object()

    def get_proxy_object(self, bus_name: str, object_path: str, introspection):
        return FakeProxy()

    def disconnect(self) -> None:
        pass


class FakeMessageBus:
    def __init__(self, *, bus_type):
        self.bus_type = bus_type

    async def connect(self):
        return FakeBus()


def test_foreground_provider_uses_async_dbus_proxy(monkeypatch) -> None:
    monkeypatch.setattr("dbus_next.aio.MessageBus", FakeMessageBus)

    foreground = ForegroundProvider().get()

    assert foreground.available is True
    assert foreground.pid == 4242
    assert foreground.app_id == "org.blender.Blender.desktop"
    assert foreground.wm_class == "Blender"
