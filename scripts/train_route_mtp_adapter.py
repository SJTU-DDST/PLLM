#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pllm.route_adapter import SharedLowRankRouteAdapter, load_route_traces
from pllm.route_mtp import RouteMTPCheckpoint
from pllm.route_mtp_torch import TorchTargetRouteHeads


DEFAULT_SLOTS = (22, 32, 64, 128, 256, 384, 448, 480, 496, 504, 512)


def request_split(items: list[Any], validation_ratio: float, seed: int) -> tuple[list[Any], list[Any]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_ratio)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def concatenate(items: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([item.features for item in items]),
        np.concatenate([item.actual_routes for item in items]),
        np.concatenate([item.mtp_routes for item in items]),
    )


def rankings_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=-1, kind="stable").astype(np.uint16)


def coverage_metrics(
    rankings: np.ndarray,
    actual_routes: np.ndarray,
    slots: tuple[int, ...],
) -> dict[str, Any]:
    if rankings.ndim != 3 or actual_routes.ndim != 3:
        raise ValueError("rankings and routes must have [sample, layer, expert] axes")
    if rankings.shape[:2] != actual_routes.shape[:2]:
        raise ValueError("rankings and routes use incompatible samples or layers")
    expert_count = rankings.shape[2]
    inverse = np.empty_like(rankings, dtype=np.uint16)
    ordinal = np.arange(expert_count, dtype=np.uint16)
    np.put_along_axis(inverse, rankings, ordinal.reshape(1, 1, -1), axis=2)
    selected_ranks = np.take_along_axis(
        inverse, actual_routes.astype(np.int64), axis=2
    ).astype(np.int32) + 1
    layer_required = selected_ranks.max(axis=2)
    token_required = layer_required.max(axis=1)
    profiles = {}
    for value in slots:
        value = min(int(value), expert_count)
        profiles[str(value)] = {
            "slots_per_layer": value,
            "all_layer_coverage": float(np.mean(token_required <= value)),
            "all_layer_miss_rate": float(np.mean(token_required > value)),
            "layer_route_coverage": float(np.mean(layer_required <= value)),
            "active_expert_recall": float(np.mean(selected_ranks <= value)),
            "resident_fraction": value / expert_count,
            "expert_weight_fraction_released": 1.0 - value / expert_count,
        }
    return {
        "samples": int(rankings.shape[0]),
        "layers": int(rankings.shape[1]),
        "experts": int(expert_count),
        "mean_token_required_uniform_rank": float(token_required.mean()),
        "p95_token_required_uniform_rank": float(np.percentile(token_required, 95)),
        "profiles": profiles,
    }


@torch.inference_mode()
def infer_adapter_rankings(
    model: SharedLowRankRouteAdapter,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    model.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        chunks.append(model(batch).argsort(dim=-1, descending=True).cpu().numpy())
    return np.concatenate(chunks).astype(np.uint16)


@torch.inference_mode()
def infer_zero_shot_rankings(
    model_path: Path,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    checkpoint = RouteMTPCheckpoint.from_model(model_path)
    heads = TorchTargetRouteHeads.from_checkpoint(
        checkpoint,
        device=str(device),
        allow_accelerator=device.type == "cuda",
    )
    chunks = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        scores = heads.forward(batch).routing_scores
        chunks.append(scores.argsort(dim=-1, descending=True).cpu().numpy())
    del heads
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks).astype(np.uint16)


def frequency_scores(actual_routes: np.ndarray, expert_count: int) -> np.ndarray:
    samples, layers, _ = actual_routes.shape
    scores = np.zeros((layers, expert_count), dtype=np.float64)
    for layer in range(layers):
        scores[layer] = np.bincount(
            actual_routes[:, layer, :].reshape(-1), minlength=expert_count
        )
    return scores


def history_rankings(
    requests: list[Any],
    global_scores: np.ndarray,
    expert_count: int,
) -> np.ndarray:
    outputs = []
    for request in requests:
        local = np.zeros_like(global_scores)
        recency = np.zeros_like(global_scores)
        for actual in request.actual_routes:
            score = global_scores + 4.0 * local + 2.0 * recency
            outputs.append(rankings_from_scores(score[None, ...])[0])
            recency *= 0.75
            for layer in range(actual.shape[0]):
                local[layer, actual[layer]] += 1.0
                recency[layer, actual[layer]] += 1.0
    return np.stack(outputs)


def mtp_frequency_rankings(
    mtp_routes: np.ndarray,
    global_scores: np.ndarray,
) -> np.ndarray:
    scores = np.broadcast_to(
        global_scores[None, :, :],
        (len(mtp_routes), *global_scores.shape),
    ).copy()
    boost = float(global_scores.max(initial=1.0) + 1.0)
    for sample, experts in enumerate(mtp_routes):
        scores[sample, :, experts] += boost
    return rankings_from_scores(scores)


def benchmark_latency(
    model: SharedLowRankRouteAdapter,
    feature: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    tensor = torch.from_numpy(feature[None, :]).to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        values = []
        for _ in range(100):
            started = time.perf_counter_ns()
            model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            values.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a layer-specific RouteMTP adapter"
    )
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("validation-ratio must be within (0, 1)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    paths = sorted(args.trace_dir.glob("trace-*.npz"))
    requests = load_route_traces(paths)
    if len(requests) < 3:
        raise RuntimeError("at least three request traces are required")
    training, validation = request_split(requests, args.validation_ratio, args.seed)
    train_features, train_routes, _ = concatenate(training)
    validation_features, validation_routes, validation_mtp = concatenate(validation)
    layer_count = train_routes.shape[1]
    active_experts = train_routes.shape[2]
    expert_count = RouteMTPCheckpoint.from_model(args.model_path).expert_count
    model = SharedLowRankRouteAdapter(
        hidden_size=train_features.shape[1],
        layer_count=layer_count,
        expert_count=expert_count,
        rank=args.rank,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    positive_weight = torch.tensor(
        (expert_count - active_experts) / active_experts,
        device=device,
    )
    history = []
    best_loss = math.inf
    best_state = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = np.random.permutation(len(train_features))
        losses = []
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            features = torch.from_numpy(train_features[indices]).to(device)
            routes = torch.from_numpy(
                train_routes[indices].astype(np.int64)
            ).to(device)
            target = torch.zeros(
                len(indices), layer_count, expert_count, device=device
            )
            target.scatter_(2, routes, 1.0)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for start in range(0, len(validation_features), args.batch_size):
                features = torch.from_numpy(
                    validation_features[start : start + args.batch_size]
                ).to(device)
                routes = torch.from_numpy(
                    validation_routes[start : start + args.batch_size].astype(np.int64)
                ).to(device)
                target = torch.zeros(
                    len(features), layer_count, expert_count, device=device
                )
                target.scatter_(2, routes, 1.0)
                validation_losses.append(
                    float(
                        F.binary_cross_entropy_with_logits(
                            model(features), target, pos_weight=positive_weight
                        ).cpu()
                    )
                )
        train_loss = float(np.mean(losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
    if best_state is None:
        raise RuntimeError("adapter training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    global_scores = frequency_scores(train_routes, expert_count)
    frequency = np.broadcast_to(
        rankings_from_scores(global_scores[None, ...]),
        (len(validation_features), layer_count, expert_count),
    )
    baselines = {
        "global_frequency": frequency,
        "request_history_frequency": history_rankings(
            validation, global_scores, expert_count
        ),
        "mtp_route_plus_frequency": mtp_frequency_rankings(
            validation_mtp, global_scores
        ),
        "original_target_gates_zero_shot": infer_zero_shot_rankings(
            args.model_path,
            validation_features,
            device,
            args.batch_size,
        ),
        "learned_layer_adapter": infer_adapter_rankings(
            model,
            validation_features,
            device,
            args.batch_size,
        ),
    }
    metrics = {
        name: coverage_metrics(rankings, validation_routes, DEFAULT_SLOTS)
        for name, rankings in baselines.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "route-mtp-adapter.pt"
    metadata = {
        "layers": list(validation[0].layers),
        "active_experts": active_experts,
        "seed": args.seed,
        "training_requests": [item.path.name for item in training],
        "validation_requests": [item.path.name for item in validation],
        "best_validation_loss": best_loss,
    }
    model.save(checkpoint_path, metadata=metadata)
    result = {
        "schema_version": 1,
        "evidence": "live_vllm_request_split_route_mtp_adapter_experiment",
        "model_path": str(args.model_path.resolve()),
        "trace_dir": str(args.trace_dir.resolve()),
        "device": str(device),
        "seed": args.seed,
        "training_requests": len(training),
        "training_samples": int(len(train_features)),
        "validation_requests": len(validation),
        "validation_samples": int(len(validation_features)),
        "split_unit": "request",
        "adapter": {
            **model.config(),
            "parameters": model.parameter_count(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "latency_batch_one": benchmark_latency(
                model, validation_features[0], device
            ),
        },
        "training": history,
        "metrics": metrics,
        "safety": {
            "target_router_remains_authoritative": True,
            "eviction_enabled": False,
            "reason": "offline shadow experiment; deployment requires held-out risk certification",
        },
        "files": metadata,
    }
    result_path = args.output_dir / "route-mtp-adapter-results.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"result": str(result_path), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
