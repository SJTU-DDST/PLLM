from __future__ import annotations

import torch

from pllm.vllm_eer_runtime import (
    flatten_cache_tensors,
    low_memory_nvfp4_scale_factor,
    release_export_layer,
    tensor_storage_bytes,
)


def _reference_scale_factor(scales: torch.Tensor, dtype: torch.dtype) -> float:
    if dtype == torch.half:
        return 1.0
    values = scales.float() * (2**7)
    positive = values[values > 0]
    if positive.numel():
        maximum = positive.max()
        if maximum < 448 * (2**7):
            return (448 * (2**7) / maximum).log2().floor().exp2().item()
    return 1.0


def test_low_memory_scale_factor_matches_vllm_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    scales = torch.rand(8192, generator=generator, dtype=torch.bfloat16)
    scales[::17] = 0

    assert low_memory_nvfp4_scale_factor(scales, torch.bfloat16) == (
        _reference_scale_factor(scales, torch.bfloat16)
    )


def test_low_memory_scale_factor_preserves_half_and_zero_cases() -> None:
    scales = torch.tensor([0.0, 0.0], dtype=torch.bfloat16)

    assert low_memory_nvfp4_scale_factor(scales, torch.bfloat16) == 1.0
    assert low_memory_nvfp4_scale_factor(scales, torch.half) == 1.0


def test_tensor_storage_bytes_supports_scalar_and_matrix() -> None:
    scalar = torch.tensor(3.25, dtype=torch.float32)
    matrix = torch.arange(12, dtype=torch.int32).reshape(3, 4)

    assert torch.frombuffer(
        bytearray(tensor_storage_bytes(scalar)), dtype=torch.float32
    ).reshape(()).item() == 3.25
    restored = torch.frombuffer(
        bytearray(tensor_storage_bytes(matrix)), dtype=torch.int32
    ).reshape(matrix.shape)
    assert torch.equal(restored, matrix)


def test_release_export_layer_drops_runtime_parameters() -> None:
    class Method:
        moe_kernel = object()
        moe_quant_config = object()

    layer = torch.nn.Module()
    layer.quant_method = Method()
    layer.register_parameter(
        "w13_weight", torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    )
    layer.workspace = torch.ones(1)

    release_export_layer(layer)

    assert "w13_weight" not in layer._parameters
    assert not hasattr(layer, "workspace")
    assert layer.quant_method.moe_kernel is None
    assert layer.quant_method.moe_quant_config is None


def test_flatten_cache_tensors_handles_hybrid_cache_layout() -> None:
    attention = torch.ones(2)
    mamba_conv = torch.ones(3)
    mamba_ssm = torch.ones(4)

    flattened = flatten_cache_tensors([attention, [mamba_conv, mamba_ssm], None])

    assert len(flattened) == 3
    assert all(
        actual is expected
        for actual, expected in zip(flattened, (attention, mamba_conv, mamba_ssm))
    )
