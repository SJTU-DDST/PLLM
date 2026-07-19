from __future__ import annotations

from types import SimpleNamespace

import pytest

from pllm.vllm_eer_runtime import validate_elastic_layer


def layer(backend: str = "marlin", monolithic: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        quant_method=SimpleNamespace(
            nvfp4_backend=SimpleNamespace(value=backend),
            is_monolithic=monolithic,
        )
    )


def test_reload_validation_does_not_require_global_vllm_config() -> None:
    validate_elastic_layer(layer(), is_reload=True)


def test_reload_validation_still_rejects_incompatible_kernel() -> None:
    with pytest.raises(RuntimeError, match="Marlin"):
        validate_elastic_layer(layer(backend="cutlass"), is_reload=True)
    with pytest.raises(RuntimeError, match="modular"):
        validate_elastic_layer(layer(monolithic=True), is_reload=True)
