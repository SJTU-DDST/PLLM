from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from pllm.vllm_eer_runtime import (
    EERRuntime,
    _ControlHandler,
    cache_storage_signature,
    validate_elastic_layer,
)


def layer(backend: str = "marlin", monolithic: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        quant_method=SimpleNamespace(
            nvfp4_backend=SimpleNamespace(value=backend),
            is_monolithic=monolithic,
        )
    )


def test_cache_storage_signature_samples_non_contiguous_views_without_copy() -> None:
    allocation = torch.arange(64, dtype=torch.uint8).reshape(8, 8)
    view = allocation[:, ::2]

    assert not view.is_contiguous()
    before = cache_storage_signature([view], sample_content=True, sample_bytes=4)
    allocation[0, 0] = 255
    after = cache_storage_signature([view], sample_content=True, sample_bytes=4)

    assert before["storage_count"] == 1
    assert before["allocated_bytes"] == allocation.untyped_storage().nbytes()
    assert before["copy_bytes"] == 12
    assert before["allocation_fingerprint"] == after["allocation_fingerprint"]
    assert before["content_sample_fingerprint"] != after["content_sample_fingerprint"]


def test_reload_validation_does_not_require_global_vllm_config() -> None:
    validate_elastic_layer(layer(), is_reload=True)


def test_reload_validation_still_rejects_incompatible_kernel() -> None:
    with pytest.raises(RuntimeError, match="Marlin"):
        validate_elastic_layer(layer(backend="cutlass"), is_reload=True)
    with pytest.raises(RuntimeError, match="modular"):
        validate_elastic_layer(layer(monolithic=True), is_reload=True)


def test_non_full_startup_profile_loads_exact_route_without_recording_it() -> None:
    class DataPlane:
        loaded: list[int] = []

        def is_fully_resident(self, _layer: int) -> bool:
            return False

        def ensure(self, _layer: int, experts: list[int], **_kwargs) -> None:
            self.loaded = experts

    class RouteWindow:
        observed = False

        def current_phase(self) -> str:
            return "idle"

        def observe_rows(self, _layer: int, _rows: list[list[int]]) -> None:
            self.observed = True

    runtime = object.__new__(EERRuntime)
    runtime.config = SimpleNamespace(mode="elastic")
    runtime.suspended = False
    runtime._startup_profile_pending = True
    runtime.data_plane = DataPlane()
    runtime.route_window = RouteWindow()
    runtime.pin_recent_steps = 32
    model_layer = SimpleNamespace(layer_name="model.layers.3.mixer.experts")

    runtime.ensure(model_layer, torch.tensor([[7, 2, 7, 5]]))

    assert runtime.data_plane.loaded == [7, 2, 5]
    assert runtime.route_window.observed is False

    runtime.finish_startup_profile()
    with pytest.raises(RuntimeError, match="only in prefill or decode"):
        runtime.ensure(model_layer, torch.tensor([[7, 2]]))


def test_non_full_prefill_loads_the_actual_route_without_decode_pins() -> None:
    class DataPlane:
        loaded: list[int] = []
        pinned: list[int] = []

        def is_fully_resident(self, _layer: int) -> bool:
            return False

        def ensure(
            self,
            _layer: int,
            experts: list[int],
            *,
            pinned_experts: list[int],
            **_kwargs,
        ) -> None:
            self.loaded = experts
            self.pinned = pinned_experts

    class RouteWindow:
        observed = False

        def current_phase(self) -> str:
            return "prefill"

        def observe_rows(self, _layer: int, _rows: list[list[int]]) -> None:
            self.observed = True

        def recent_experts(self, _layer: int, _steps: int) -> list[int]:
            return [511]

    runtime = object.__new__(EERRuntime)
    runtime.config = SimpleNamespace(mode="elastic")
    runtime.suspended = False
    runtime._startup_profile_pending = False
    runtime.data_plane = DataPlane()
    runtime.route_window = RouteWindow()
    runtime.pin_recent_steps = 32
    model_layer = SimpleNamespace(layer_name="model.layers.3.mixer.experts")

    runtime.ensure(model_layer, torch.tensor([[7, 2, 7, 5]]))

    assert runtime.data_plane.loaded == [7, 2, 5]
    assert runtime.data_plane.pinned == []
    assert runtime.route_window.observed is True


def test_control_handler_ignores_client_disconnect_while_writing_response() -> None:
    class DisconnectedWriter:
        def write(self, _payload: bytes) -> None:
            raise BrokenPipeError(32, "broken pipe")

    runtime = SimpleNamespace(
        last_error="",
        handle_command=lambda _request: {"state": "loading"},
    )
    handler = object.__new__(_ControlHandler)
    handler.server = SimpleNamespace(runtime=runtime)
    handler.rfile = BytesIO(b'{"command":"status"}\n')
    handler.wfile = DisconnectedWriter()

    handler.handle()

    assert runtime.last_error == ""
