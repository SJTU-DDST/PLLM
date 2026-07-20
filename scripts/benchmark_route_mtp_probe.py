#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from pllm.config import DEFAULT_MODEL_PATH
from pllm.route_mtp import RouteMTPCheckpoint
from pllm.route_mtp_torch import (
    RouteMTPAttentionState,
    TorchRouteMTPProbe,
    TorchTargetRouteHeads,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU microbenchmark for the 136MiB Nemotron RouteMTP gate probe"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cache-reset-steps", type=int, default=32)
    parser.add_argument(
        "--target-route-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.cache_reset_steps <= 0:
        parser.error("step counts must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        parser.error("CUDA is not available")

    checkpoint = RouteMTPCheckpoint.from_model(args.model_path)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    process_allocated_before = (
        torch.cuda.memory_allocated() if args.device.startswith("cuda") else 0
    )
    before_free, before_total = (
        torch.cuda.mem_get_info() if args.device.startswith("cuda") else (0, 0)
    )
    probe = TorchRouteMTPProbe.from_checkpoint(
        checkpoint,
        device=args.device,
        allow_accelerator=args.device != "cpu",
    )
    process_allocated_after_load = (
        torch.cuda.memory_allocated() if args.device.startswith("cuda") else 0
    )
    target_heads = (
        TorchTargetRouteHeads.from_checkpoint(
            checkpoint,
            device=args.device,
            allow_accelerator=args.device != "cpu",
        )
        if args.target_route_heads
        else None
    )
    process_allocated_after_target_heads = (
        torch.cuda.memory_allocated() if args.device.startswith("cuda") else 0
    )
    generator = torch.Generator(device=args.device).manual_seed(20260719)
    embedding = torch.randn(
        1,
        checkpoint.hidden_size,
        dtype=probe.dtype,
        device=args.device,
        generator=generator,
    )
    hidden = torch.randn(
        1,
        checkpoint.hidden_size,
        dtype=probe.dtype,
        device=args.device,
        generator=generator,
    )
    first = probe.forward(embedding, hidden, RouteMTPAttentionState())
    repeated = probe.forward(embedding, hidden, RouteMTPAttentionState())
    deterministic = torch.equal(first.topk_experts, repeated.topk_experts)
    unique_topk = len(set(first.topk_experts[0].tolist()))
    target_output = (
        target_heads.forward(first.route_hidden) if target_heads is not None else None
    )
    from vllm import _custom_ops as vllm_ops

    norm_weight = probe.weights["mtp.layers.1.norm.weight"]
    official_plain = torch.empty_like(hidden)
    vllm_ops.rms_norm(
        official_plain, hidden, norm_weight, checkpoint.norm_epsilon
    )
    probe_plain = probe._rms_norm(
        hidden, norm_weight
    )
    official_fused = hidden.clone()
    official_residual = embedding.clone()
    vllm_ops.fused_add_rms_norm(
        official_fused,
        official_residual,
        norm_weight,
        checkpoint.norm_epsilon,
    )
    expected_residual = hidden + embedding
    probe_fused = probe._rms_norm(
        expected_residual, norm_weight
    )
    rms_plain_max_abs_error = float(
        (official_plain - probe_plain).abs().max().item()
    )
    rms_fused_max_abs_error = float(
        (official_fused - probe_fused).abs().max().item()
    )
    residual_max_abs_error = float(
        (official_residual - expected_residual).abs().max().item()
    )

    state = RouteMTPAttentionState()
    for _ in range(args.warmup):
        state = probe.forward(embedding, hidden, state).state
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    timings = []
    state = RouteMTPAttentionState()
    for step in range(args.steps):
        if step and step % args.cache_reset_steps == 0:
            state = RouteMTPAttentionState()
        started = time.perf_counter_ns()
        output = probe.forward(embedding, hidden, state)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - started) / 1e6)
        state = output.state

    route_head_timings = []
    combined_timings = []
    shadow_materialization_timings = []
    materialized_score_shape = None
    if target_heads is not None:
        for _ in range(args.steps):
            started = time.perf_counter_ns()
            target_heads.forward(first.route_hidden)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            route_head_timings.append((time.perf_counter_ns() - started) / 1e6)
        state = RouteMTPAttentionState()
        for step in range(args.steps):
            if step and step % args.cache_reset_steps == 0:
                state = RouteMTPAttentionState()
            started = time.perf_counter_ns()
            combined = probe.forward(embedding, hidden, state)
            target_heads.forward(combined.route_hidden)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            combined_timings.append((time.perf_counter_ns() - started) / 1e6)
            state = combined.state
        state = RouteMTPAttentionState()
        for step in range(args.steps):
            if step and step % args.cache_reset_steps == 0:
                state = RouteMTPAttentionState()
            started = time.perf_counter_ns()
            combined = probe.forward(embedding, hidden, state)
            target = target_heads.forward(combined.route_hidden)
            score_rows = (
                target.routing_scores[0]
                .to(dtype=torch.float16)
                .cpu()
                .numpy()
            )
            target.topk_experts[0].cpu().numpy()
            combined.topk_experts[0].tolist()
            shadow_materialization_timings.append(
                (time.perf_counter_ns() - started) / 1e6
            )
            materialized_score_shape = list(score_rows.shape)
            state = combined.state

    after_free, _after_total = (
        torch.cuda.mem_get_info() if args.device.startswith("cuda") else (0, 0)
    )
    process_allocated_after_run = (
        torch.cuda.memory_allocated() if args.device.startswith("cuda") else 0
    )
    process_peak_allocated = (
        torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    )
    result = {
        "schema_version": 1,
        "evidence": "real_checkpoint_route_mtp_probe_gpu_forward",
        "model_path": str(checkpoint.model_path),
        "device": args.device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda")
            else ""
        ),
        "steps": args.steps,
        "warmup": args.warmup,
        "probe_tensor_bytes": checkpoint.weight_plan().selected_bytes,
        "probe_storage_bytes": probe.allocated_bytes(),
        "free_gpu_bytes_before": before_free,
        "free_gpu_bytes_after": after_free,
        "gpu_bytes_delta": before_free - after_free,
        "gpu_bytes_delta_scope": "global_device_confounded_by_other_processes",
        "process_cuda_allocated_before": process_allocated_before,
        "process_cuda_allocated_after_load": process_allocated_after_load,
        "process_cuda_allocated_after_target_heads": process_allocated_after_target_heads,
        "process_cuda_allocated_after_run": process_allocated_after_run,
        "process_cuda_peak_allocated": process_peak_allocated,
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_p50": percentile(timings, 0.50),
        "latency_ms_p95": percentile(timings, 0.95),
        "latency_ms_max": max(timings),
        "target_route_head_latency_ms_mean": (
            statistics.fmean(route_head_timings) if route_head_timings else None
        ),
        "target_route_head_latency_ms_p95": (
            percentile(route_head_timings, 0.95) if route_head_timings else None
        ),
        "combined_latency_ms_mean": (
            statistics.fmean(combined_timings) if combined_timings else None
        ),
        "combined_latency_ms_p95": (
            percentile(combined_timings, 0.95) if combined_timings else None
        ),
        "shadow_materialization_latency_ms_mean": (
            statistics.fmean(shadow_materialization_timings)
            if shadow_materialization_timings
            else None
        ),
        "shadow_materialization_latency_ms_p95": (
            percentile(shadow_materialization_timings, 0.95)
            if shadow_materialization_timings
            else None
        ),
        "materialized_score_shape": materialized_score_shape,
        "materialized_score_dtype": "float16" if materialized_score_shape else None,
        "deterministic_fresh_state": deterministic,
        "unique_topk": unique_topk,
        "topk": first.topk_experts[0].tolist(),
        "router_logits_finite": bool(torch.isfinite(first.router_logits).all()),
        "vllm_rms_plain_max_abs_error": rms_plain_max_abs_error,
        "vllm_rms_fused_max_abs_error": rms_fused_max_abs_error,
        "vllm_residual_max_abs_error": residual_max_abs_error,
        "attention_cache_tokens": state.tokens,
        "target_route_head_checkpoint_bytes": sum(
            item.size_bytes for item in checkpoint.target_gate_tensors
        ),
        "target_route_head_runtime_bytes": (
            target_heads.allocated_bytes() if target_heads is not None else 0
        ),
        "target_route_layers": (
            list(target_output.layers) if target_output is not None else []
        ),
        "target_route_topk_unique_per_layer": (
            [
                int(target_output.topk_experts[0, index].unique().numel())
                for index in range(len(target_output.layers))
            ]
            if target_output is not None
            else []
        ),
        "target_router_logits_finite": (
            bool(torch.isfinite(target_output.router_logits).all())
            if target_output is not None
            else None
        ),
        "target_layer_routes_produced": target_output is not None,
        "target_layer_routes_calibrated": False,
        "mtp_experts_executed": False,
    }
    if before_total:
        result["gpu_total_bytes"] = before_total
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    target_valid = bool(
        target_output is None
        or (
            torch.isfinite(target_output.router_logits).all()
            and torch.isfinite(target_output.routing_scores).all()
            and all(
                target_output.topk_experts[0, index].unique().numel()
                == checkpoint.active_experts
                for index in range(len(target_output.layers))
            )
        )
    )
    return 0 if deterministic and unique_topk == checkpoint.active_experts and target_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
