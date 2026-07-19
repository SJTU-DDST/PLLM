from __future__ import annotations

import math
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable


VALID_PHASES = {"idle", "prefill", "decode"}


@dataclass(slots=True, frozen=True)
class DecodeResidencyDecision:
    action: str
    slots_per_layer: int
    projected_byte_hit_rate: float
    miss_gib_per_token: float
    miss_gib_s: float
    estimated_slowdown_ratio: float
    reason: str
    misses_per_token: float = 0.0
    miss_latency_ms_per_token: float = 0.0
    evidence: str = "live_decode_route_window"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class LayerResidencyDecision:
    action: str
    slots_per_layer: int
    slots_by_layer: dict[int, int]
    projected_byte_hit_rate: float
    miss_gib_per_token: float
    miss_gib_s: float
    estimated_slowdown_ratio: float
    reason: str
    projected_reclaim_bytes: int = 0
    shrink_copy_bytes: int = 0
    immediate_expand_bytes: int = 0
    future_expand_bytes: int = 0
    immediate_transition_seconds: float = 0.0
    future_transition_seconds: float = 0.0
    transition_seconds: float = 0.0
    amortized_transition_ms_per_token: float = 0.0
    miss_latency_ms_per_token: float = 0.0
    remaining_decode_tokens: int = 0
    prediction_windows: int = 0
    planner_wall_ms: float = 0.0
    planner_peak_states: int = 0
    risk_estimator: str = "sum_layer_worst_observed_miss_at_source_batch_p95"
    evidence: str = "heldout_next_window_route_prediction"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slots_by_layer"] = {
            str(layer): slots for layer, slots in self.slots_by_layer.items()
        }
        return payload


class DecodeRouteWindow:
    """Past-window route predictor with held-out next-window validation.

    A completed window ranks experts for the following window.  Hit-rate and
    miss-tail estimates are recorded only after that following window closes,
    avoiding the optimistic same-window coverage used by the first prototype.
    """

    def __init__(
        self,
        layers: Iterable[int],
        experts_per_layer: int,
        window_steps: int = 256,
        validation_profiles: Iterable[int] = (
            64,
            128,
            256,
            320,
            384,
            448,
            480,
            496,
            504,
            512,
        ),
    ) -> None:
        if experts_per_layer <= 0 or window_steps <= 0:
            raise ValueError("experts_per_layer and window_steps must be positive")
        self.layers = tuple(sorted(int(layer) for layer in layers))
        if not self.layers:
            raise ValueError("at least one MoE layer is required")
        self.experts_per_layer = int(experts_per_layer)
        self.window_steps = int(window_steps)
        self.validation_profiles = tuple(
            sorted(
                {
                    max(1, min(int(profile), self.experts_per_layer))
                    for profile in validation_profiles
                }
            )
        )
        self.phase = "idle"
        self.phase_changed_at = time.time()
        self._request_generation = 0
        self._route_generation = 0
        self._current: dict[int, list[tuple[int, ...]]] = {
            layer: [] for layer in self.layers
        }
        self._recent: dict[int, deque[tuple[int, ...]]] = {
            layer: deque(maxlen=self.window_steps) for layer in self.layers
        }
        self._prefill_tail: dict[int, deque[tuple[int, ...]]] = {
            layer: deque(maxlen=min(16, self.window_steps)) for layer in self.layers
        }
        self._prediction_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self._prediction_recency: dict[int, dict[int, int]] = defaultdict(dict)
        self._prefill_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self._completed_windows = Counter()
        self._validation: dict[int, dict[int, deque[dict[str, float]]]] = {
            layer: defaultdict(lambda: deque(maxlen=8)) for layer in self.layers
        }
        self._observations = Counter()
        self._tokens = Counter()
        self._lock = threading.RLock()

    def set_phase(self, phase: str, reset_decode: bool = False) -> None:
        normalized = phase.strip().lower()
        if normalized not in VALID_PHASES:
            raise ValueError(f"invalid inference phase: {phase}")
        with self._lock:
            if normalized != self.phase:
                self.phase = normalized
                self.phase_changed_at = time.time()
            if reset_decode:
                self._request_generation += 1
                self._route_generation += 1
                for layer in self.layers:
                    self._current[layer].clear()
                    self._recent[layer].clear()
                    self._prediction_counts[layer].clear()
                    self._prediction_recency[layer].clear()
                    self._validation[layer].clear()
                    self._completed_windows[layer] = 0
                    self._prefill_tail[layer].clear()
                    self._prefill_counts[layer].clear()
                self._observations["decode"] = 0
                self._tokens["decode"] = 0

    def current_phase(self) -> str:
        with self._lock:
            return self.phase

    def observe(
        self,
        layer: int,
        experts: Iterable[int],
        token_count: int = 1,
    ) -> None:
        layer = int(layer)
        unique = tuple(dict.fromkeys(int(expert) for expert in experts))
        if layer not in self._current:
            raise KeyError(f"layer {layer} is not tracked")
        if not unique:
            return
        if any(expert < 0 or expert >= self.experts_per_layer for expert in unique):
            raise ValueError("route observation contains an out-of-range expert")
        with self._lock:
            phase = self.phase
            self._observations[phase] += 1
            self._tokens[phase] += max(1, int(token_count))
            if phase == "decode":
                for _ in range(max(1, int(token_count))):
                    self._observe_decode_row(layer, unique)
            elif phase == "prefill":
                rows = self._prefill_tail[layer]
                if len(rows) == rows.maxlen:
                    self._prefill_counts[layer].subtract(rows[0])
                    self._prefill_counts[layer] += Counter()
                rows.append(unique)
                self._prefill_counts[layer].update(unique)

    def observe_rows(self, layer: int, rows: Iterable[Iterable[int]]) -> None:
        """Record exact per-token Top-k rows without collapsing fused batches."""
        layer = int(layer)
        if layer not in self._current:
            raise KeyError(f"layer {layer} is not tracked")
        normalized: list[tuple[int, ...]] = []
        for row in rows:
            values = tuple(int(expert) for expert in row if int(expert) >= 0)
            if not values:
                continue
            if any(expert >= self.experts_per_layer for expert in values):
                raise ValueError("route observation contains an out-of-range expert")
            normalized.append(values)
        if not normalized:
            return
        with self._lock:
            phase = self.phase
            self._observations[phase] += 1
            self._tokens[phase] += len(normalized)
            if phase == "decode":
                for row in normalized:
                    self._observe_decode_row(layer, row)
            elif phase == "prefill":
                for row in normalized[-self._prefill_tail[layer].maxlen :]:
                    tail = self._prefill_tail[layer]
                    if len(tail) == tail.maxlen:
                        self._prefill_counts[layer].subtract(tail[0])
                        self._prefill_counts[layer] += Counter()
                    tail.append(row)
                    self._prefill_counts[layer].update(row)

    def _observe_decode_row(self, layer: int, row: tuple[int, ...]) -> None:
        self._recent[layer].append(row)
        current = self._current[layer]
        current.append(row)
        if len(current) >= self.window_steps:
            self._seal_window(layer, current[: self.window_steps])
            del current[: self.window_steps]

    def _seal_window(self, layer: int, rows: list[tuple[int, ...]]) -> None:
        previous = self._prediction_counts.get(layer, Counter())
        if previous:
            for slots in self.validation_profiles:
                hot = set(self._rank_counts(layer, previous)[:slots])
                misses = [sum(expert not in hot for expert in row) for row in rows]
                accesses = sum(len(row) for row in rows)
                total_misses = sum(misses)
                ordered = sorted(misses)
                p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
                self._validation[layer][slots].append(
                    {
                        "byte_hit_rate": (
                            (accesses - total_misses) / accesses if accesses else 0.0
                        ),
                        "mean_misses_per_token": (
                            total_misses / len(rows) if rows else 0.0
                        ),
                        "p95_misses_per_token": (
                            float(ordered[p95_index]) if ordered else 0.0
                        ),
                        "max_misses_per_token": (
                            float(ordered[-1]) if ordered else 0.0
                        ),
                    }
                )
        counts = Counter(expert for row in rows for expert in row)
        recency: dict[int, int] = {}
        for position, row in enumerate(rows, start=1):
            for expert in row:
                recency[expert] = position
        self._prediction_counts[layer] = counts
        self._prediction_recency[layer] = recency
        self._completed_windows[layer] += 1
        self._route_generation += 1

    def recent_experts(self, layer: int, steps: int) -> list[int]:
        if steps <= 0:
            return []
        with self._lock:
            recent: list[int] = []
            seen: set[int] = set()
            rows = list(self._recent.get(int(layer), ()))
            for row in reversed(rows[-int(steps) :]):
                for expert in row:
                    if expert not in seen:
                        recent.append(expert)
                        seen.add(expert)
            return recent

    def hot_experts(
        self, layer: int, limit: int, pin_recent_steps: int = 0
    ) -> list[int]:
        if limit <= 0:
            return []
        with self._lock:
            decode_counts = self._prediction_counts.get(layer, Counter())
            prefill_counts = self._prefill_counts.get(layer, Counter())
            recent = self._prediction_recency.get(layer, {})
            ranking = sorted(
                range(self.experts_per_layer),
                key=lambda expert: (
                    -decode_counts.get(expert, 0),
                    -prefill_counts.get(expert, 0),
                    -recent.get(expert, 0),
                    expert,
                ),
            )
            pinned = self.recent_experts(layer, pin_recent_steps)
            selected = pinned + [expert for expert in ranking if expert not in pinned]
            return selected[: min(int(limit), self.experts_per_layer)]

    def _rank_counts(self, layer: int, counts: Counter[int]) -> list[int]:
        prefill_counts = self._prefill_counts.get(layer, Counter())
        recent = self._prediction_recency.get(layer, {})
        return sorted(
            range(self.experts_per_layer),
            key=lambda expert: (
                -counts.get(expert, 0),
                -prefill_counts.get(expert, 0),
                -recent.get(expert, 0),
                expert,
            ),
        )

    def projected_hit_rate(self, slots_per_layer: int) -> float:
        slots = max(0, min(int(slots_per_layer), self.experts_per_layer))
        if slots >= self.experts_per_layer:
            return 1.0
        with self._lock:
            hits = 0
            accesses = 0
            for layer in self.layers:
                counts = self._prediction_counts.get(layer, Counter())
                if not counts:
                    continue
                hot = set(self.hot_experts(layer, slots))
                hits += sum(count for expert, count in counts.items() if expert in hot)
                accesses += sum(counts.values())
            return hits / accesses if accesses else 0.0

    def next_window_profiles(self, profiles: Iterable[int]) -> dict[str, Any]:
        """Return conservative metrics learned on previous->current transitions."""
        selected = sorted(
            {
                max(1, min(int(profile), self.experts_per_layer))
                for profile in profiles
            }
        )
        per_profile: dict[str, Any] = {}
        for slots in selected:
            per_layer: dict[str, Any] = {}
            total_hits = 0.0
            total_accesses = 0.0
            heldout_windows = []
            for layer in self.layers:
                history = list(self._validation[layer].get(slots, ()))
                if slots >= self.experts_per_layer:
                    metrics = {
                        "byte_hit_rate_lower_bound": 1.0,
                        "mean_misses_per_token_upper_bound": 0.0,
                        "p95_misses_per_token_upper_bound": 0.0,
                        "max_misses_per_token_upper_bound": 0.0,
                        "heldout_windows": int(self._completed_windows[layer]),
                    }
                elif history:
                    metrics = {
                        "byte_hit_rate_lower_bound": min(
                            item["byte_hit_rate"] for item in history
                        ),
                        "mean_misses_per_token_upper_bound": max(
                            item["mean_misses_per_token"] for item in history
                        ),
                        "p95_misses_per_token_upper_bound": max(
                            item["p95_misses_per_token"] for item in history
                        ),
                        "max_misses_per_token_upper_bound": max(
                            item["max_misses_per_token"] for item in history
                        ),
                        "heldout_windows": len(history),
                    }
                else:
                    metrics = {
                        "byte_hit_rate_lower_bound": 0.0,
                        "mean_misses_per_token_upper_bound": float("inf"),
                        "p95_misses_per_token_upper_bound": float("inf"),
                        "max_misses_per_token_upper_bound": float("inf"),
                        "heldout_windows": 0,
                    }
                per_layer[str(layer)] = metrics
                heldout_windows.append(int(metrics["heldout_windows"]))
                if math.isfinite(metrics["mean_misses_per_token_upper_bound"]):
                    accesses = 1.0
                    total_hits += float(metrics["byte_hit_rate_lower_bound"]) * accesses
                    total_accesses += accesses
            per_profile[str(slots)] = {
                "byte_hit_rate_lower_bound": (
                    total_hits / total_accesses if total_accesses else 0.0
                ),
                "heldout_windows": min(heldout_windows, default=0),
                "per_layer": per_layer,
            }
        completed = [int(self._completed_windows[layer]) for layer in self.layers]
        return {
            "prediction_ready": bool(completed) and min(completed) >= 2,
            "minimum_completed_windows": min(completed, default=0),
            "request_generation": self._request_generation,
            "route_generation": self._route_generation,
            "profiles": per_profile,
            "estimator": "past_window_rank_heldout_on_next_window",
            "risk_bound": "worst_observed_transition_window",
        }

    def status(
        self,
        profiles: Iterable[int] = (
            64,
            128,
            256,
            320,
            384,
            448,
            480,
            496,
            504,
            512,
        ),
    ) -> dict[str, Any]:
        with self._lock:
            decode_rows = sum(len(rows) for rows in self._current.values())
            prefill_rows = sum(len(rows) for rows in self._prefill_tail.values())
            projected = {
                str(profile): round(self.projected_hit_rate(profile), 6)
                for profile in profiles
                if 0 < int(profile) <= self.experts_per_layer
            }
            return {
                "phase": self.phase,
                "phase_changed_at": self.phase_changed_at,
                "request_generation": self._request_generation,
                "route_generation": self._route_generation,
                "window_steps": self.window_steps,
                "decode_observations": int(self._observations["decode"]),
                "decode_token_rows": int(self._tokens["decode"]),
                "decode_layer_rows_retained": decode_rows,
                "prefill_layer_rows_retained": prefill_rows,
                "projected_byte_hit_rate": projected,
                "storage": "cpu_ring",
                "gpu_persistent_bytes": 0,
                "next_window": self.next_window_profiles(profiles),
                "evidence": "runtime_actual_topk_per_token_past_to_next_window",
            }


class DecodeResidencyGuardrail:
    """Rejects elastic plans that predict order-of-magnitude latency loss."""

    def __init__(
        self,
        all_miss_gib_per_token: float,
        all_miss_objects_per_token: int = 880,
        miss_latency_p95_ms: float = 7.5,
        minimum_byte_hit_rate: float = 0.95,
        maximum_slowdown_ratio: float = 5.0,
    ) -> None:
        if (
            all_miss_gib_per_token <= 0
            or all_miss_objects_per_token <= 0
            or miss_latency_p95_ms <= 0
        ):
            raise ValueError("miss bytes, object count, and latency must be positive")
        if not 0 <= minimum_byte_hit_rate <= 1:
            raise ValueError("minimum_byte_hit_rate must be within [0, 1]")
        if not 1 <= maximum_slowdown_ratio < 10:
            raise ValueError("maximum_slowdown_ratio must be within [1, 10)")
        self.all_miss_gib_per_token = float(all_miss_gib_per_token)
        self.all_miss_objects_per_token = int(all_miss_objects_per_token)
        self.miss_latency_p95_ms = float(miss_latency_p95_ms)
        self.minimum_byte_hit_rate = float(minimum_byte_hit_rate)
        self.maximum_slowdown_ratio = float(maximum_slowdown_ratio)

    def choose(
        self,
        phase: str,
        projected_hit_rates: dict[int, float],
        candidate_slots: Iterable[int],
        io_budget_gib_s: float,
        token_rate: float,
        baseline_tpot_ms: float,
        full_slots: int = 512,
    ) -> DecodeResidencyDecision:
        if phase != "decode":
            return DecodeResidencyDecision(
                "full_resident",
                full_slots,
                1.0,
                0.0,
                0.0,
                1.0,
                "elastic eviction is forbidden outside decode",
            )
        if io_budget_gib_s <= 0 or token_rate <= 0 or baseline_tpot_ms <= 0:
            raise ValueError("I/O budget, token rate, and baseline TPOT must be positive")

        baseline_seconds = baseline_tpot_ms / 1000.0
        candidates = sorted(
            {int(slots) for slots in candidate_slots if 0 < int(slots) < full_slots}
        )
        rejected: list[str] = []
        for slots in candidates:
            hit_rate = max(0.0, min(1.0, float(projected_hit_rates.get(slots, 0.0))))
            miss_gib = self.all_miss_gib_per_token * (1.0 - hit_rate)
            misses = self.all_miss_objects_per_token * (1.0 - hit_rate)
            miss_gib_s = miss_gib * token_rate
            fixed_latency_seconds = misses * self.miss_latency_p95_ms / 1000.0
            miss_seconds = max(
                miss_gib / io_budget_gib_s,
                fixed_latency_seconds,
            )
            slowdown = (baseline_seconds + miss_seconds) / baseline_seconds
            if hit_rate < self.minimum_byte_hit_rate:
                rejected.append(f"{slots}:hit={hit_rate:.3f}")
                continue
            if miss_gib_s > io_budget_gib_s:
                rejected.append(f"{slots}:io={miss_gib_s:.3f}GiB/s")
                continue
            if slowdown >= min(10.0, self.maximum_slowdown_ratio):
                rejected.append(f"{slots}:slowdown={slowdown:.2f}x")
                continue
            return DecodeResidencyDecision(
                "decode_elastic",
                slots,
                hit_rate,
                miss_gib,
                miss_gib_s,
                slowdown,
                "decode route window satisfies hit, I/O, and latency guardrails",
                misses,
                miss_seconds * 1000.0,
            )

        return DecodeResidencyDecision(
            "yield",
            full_slots,
            max(projected_hit_rates.values(), default=0.0),
            self.all_miss_gib_per_token,
            self.all_miss_gib_per_token * token_rate,
            math.inf,
            "no elastic profile satisfies guardrails: " + ", ".join(rejected),
            self.all_miss_objects_per_token,
            self.all_miss_objects_per_token * self.miss_latency_p95_ms,
        )


class HorizonAwareLayerPlanner:
    """Multiple-choice capacity planner over independently sized MoE layers.

    The planner consumes only held-out past->next-window measurements.  It
    includes immediate compaction, future full-prefill restoration, exact miss
    traffic, and per-layer miss-tail latency in the request wall-time bound.
    """

    def __init__(
        self,
        *,
        top_k: int,
        minimum_byte_hit_rate: float = 0.95,
        maximum_slowdown_ratio: float = 5.0,
        minimum_heldout_windows: int = 1,
        miss_latency_curve_ms: dict[int, float] | None = None,
        reclaim_bucket_bytes: int = 16 * 1024**2,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= minimum_byte_hit_rate <= 1:
            raise ValueError("minimum_byte_hit_rate must be within [0, 1]")
        if not 1 <= maximum_slowdown_ratio < 10:
            raise ValueError("maximum_slowdown_ratio must be within [1, 10)")
        if minimum_heldout_windows <= 0 or reclaim_bucket_bytes <= 0:
            raise ValueError("window and reclaim bucket sizes must be positive")
        curve = miss_latency_curve_ms or {1: 7.5, top_k: 113.0}
        if not curve or any(int(key) <= 0 or float(value) <= 0 for key, value in curve.items()):
            raise ValueError("miss latency curve must contain positive samples")
        self.top_k = int(top_k)
        self.minimum_byte_hit_rate = float(minimum_byte_hit_rate)
        self.maximum_slowdown_ratio = float(maximum_slowdown_ratio)
        self.minimum_heldout_windows = int(minimum_heldout_windows)
        self.miss_latency_curve_ms = {
            int(key): float(value) for key, value in curve.items()
        }
        self.reclaim_bucket_bytes = int(reclaim_bucket_bytes)

    def choose(
        self,
        *,
        phase: str,
        prediction: dict[str, Any],
        candidate_slots: Iterable[int],
        layer_bytes: dict[int, int],
        current_slots_by_layer: dict[int, int],
        target_reclaim_bytes: int,
        io_budget_gib_s: float,
        token_rate: float,
        baseline_tpot_ms: float,
        remaining_decode_tokens: int,
        resize_copy_gib_s: float,
        expand_gib_s: float,
        rebuild_ms_per_layer: float,
        release_deadline_ms: float,
        full_slots: int = 512,
    ) -> LayerResidencyDecision:
        planner_started = time.perf_counter()
        layers = sorted(int(layer) for layer in layer_bytes)
        full = {layer: int(full_slots) for layer in layers}
        if phase != "decode":
            return self._fallback(
                "full_resident",
                full,
                remaining_decode_tokens,
                "elastic eviction is forbidden outside decode",
            )
        if target_reclaim_bytes <= 0:
            return self._fallback(
                "full_resident",
                full,
                remaining_decode_tokens,
                "foreground capacity envelope does not require expert release",
            )
        if not bool(prediction.get("prediction_ready")):
            return self._fallback(
                "observe",
                full,
                remaining_decode_tokens,
                "two completed windows are required for held-out next-window prediction",
            )
        if remaining_decode_tokens <= 0:
            return self._fallback(
                "yield",
                full,
                remaining_decode_tokens,
                "remaining decode horizon is unknown; resize cannot be amortized",
            )
        positive = (
            io_budget_gib_s,
            token_rate,
            baseline_tpot_ms,
            resize_copy_gib_s,
            expand_gib_s,
            release_deadline_ms,
        )
        if any(value <= 0 for value in positive) or rebuild_ms_per_layer < 0:
            raise ValueError("planner bandwidth, latency, and horizon inputs are invalid")

        profiles = dict(prediction.get("profiles") or {})
        candidates = sorted(
            {
                int(slots)
                for slots in candidate_slots
                if self.top_k <= int(slots) < full_slots
            }
            | {int(full_slots)}
        )
        maximum_bucket = math.ceil(
            sum(layer_bytes.values()) / self.reclaim_bucket_bytes
        )
        states: dict[int, list[dict[str, Any]]] = {
            0: [{
                "objective_ms": 0.0,
                "parent": None,
                "option": None,
                "reclaim": 0,
                "mean_misses": 0.0,
                "miss_bytes": 0.0,
                "miss_latency_ms": 0.0,
                "shrink_copy": 0,
                "immediate_expand": 0,
                "future_expand": 0,
                "immediate_seconds": 0.0,
                "future_seconds": 0.0,
                "transition_seconds": 0.0,
                "minimum_windows": math.inf,
            }]
        }
        peak_states = 1
        for layer in layers:
            size = int(layer_bytes[layer])
            current = int(current_slots_by_layer.get(layer, full_slots))
            options: list[dict[str, Any]] = []
            for slots in candidates:
                if slots == full_slots:
                    hit_rate = 1.0
                    mean_misses = 0.0
                    risk_misses = 0.0
                    windows = int(prediction.get("minimum_completed_windows", 0))
                else:
                    layer_metric = (
                        dict(profiles.get(str(slots), {}))
                        .get("per_layer", {})
                        .get(str(layer), {})
                    )
                    windows = int(layer_metric.get("heldout_windows", 0))
                    if windows < self.minimum_heldout_windows:
                        continue
                    hit_rate = float(
                        layer_metric.get("byte_hit_rate_lower_bound", 0.0)
                    )
                    mean_misses = float(
                        layer_metric.get("mean_misses_per_token_upper_bound", math.inf)
                    )
                    risk_misses = float(
                        layer_metric.get(
                            "max_misses_per_token_upper_bound",
                            layer_metric.get(
                                "p95_misses_per_token_upper_bound", math.inf
                            ),
                        )
                    )
                    if not math.isfinite(mean_misses) or not math.isfinite(risk_misses):
                        continue
                reclaim = int(round(size * (full_slots - slots) / full_slots))
                object_bytes = size / full_slots
                miss_bytes = mean_misses * object_bytes
                miss_latency_ms = self._batch_latency_ms(risk_misses)
                changed = slots != current
                shrink_copy = (
                    int(round(size * slots / full_slots))
                    if changed and slots < current
                    else 0
                )
                immediate_expand = (
                    int(round(size * slots / full_slots))
                    if changed and slots > current
                    else 0
                )
                future_expand = size if slots < full_slots else 0
                immediate_seconds = (
                    shrink_copy / (resize_copy_gib_s * 1024**3)
                    + immediate_expand / (expand_gib_s * 1024**3)
                    + int(changed) * rebuild_ms_per_layer / 1000.0
                )
                future_seconds = (
                    future_expand / (expand_gib_s * 1024**3)
                    + int(slots < full_slots) * rebuild_ms_per_layer / 1000.0
                )
                transition_seconds = immediate_seconds + future_seconds
                objective = (
                    miss_latency_ms
                    + transition_seconds * 1000.0 / remaining_decode_tokens
                )
                options.append(
                    {
                        "layer": layer,
                        "slots": slots,
                        "hit_rate": hit_rate,
                        "mean_misses": mean_misses,
                        "miss_bytes": miss_bytes,
                        "miss_latency_ms": miss_latency_ms,
                        "reclaim": reclaim,
                        "shrink_copy": shrink_copy,
                        "immediate_expand": immediate_expand,
                        "future_expand": future_expand,
                        "immediate_seconds": immediate_seconds,
                        "future_seconds": future_seconds,
                        "transition_seconds": transition_seconds,
                        "objective_ms": objective,
                        "windows": windows,
                    }
                )
            if not options:
                return self._fallback(
                    "yield",
                    full,
                    remaining_decode_tokens,
                    f"layer {layer} has no validated residency option",
                )
            next_states: dict[int, list[dict[str, Any]]] = {}
            for frontier in states.values():
                for state in frontier:
                    for option in options:
                        reclaim = int(state["reclaim"]) + int(option["reclaim"])
                        bucket = min(
                            math.ceil(reclaim / self.reclaim_bucket_bytes),
                            maximum_bucket,
                        )
                        objective = (
                            float(state["objective_ms"]) + option["objective_ms"]
                        )
                        candidate = {
                            "objective_ms": objective,
                            "parent": state,
                            "option": option,
                            "reclaim": reclaim,
                            "mean_misses": state["mean_misses"]
                            + option["mean_misses"],
                            "miss_bytes": state["miss_bytes"]
                            + option["miss_bytes"],
                            "miss_latency_ms": state["miss_latency_ms"]
                            + option["miss_latency_ms"],
                            "shrink_copy": state["shrink_copy"]
                            + option["shrink_copy"],
                            "immediate_expand": state["immediate_expand"]
                            + option["immediate_expand"],
                            "future_expand": state["future_expand"]
                            + option["future_expand"],
                            "immediate_seconds": state["immediate_seconds"]
                            + option["immediate_seconds"],
                            "future_seconds": state["future_seconds"]
                            + option["future_seconds"],
                            "transition_seconds": state["transition_seconds"]
                            + option["transition_seconds"],
                            "minimum_windows": min(
                                state["minimum_windows"], option["windows"]
                            ),
                        }
                        self._insert_frontier(
                            next_states.setdefault(bucket, []), candidate
                        )
            states = next_states
            peak_states = max(
                peak_states, sum(len(frontier) for frontier in states.values())
            )

        feasible: list[tuple[float, int, LayerResidencyDecision]] = []
        for frontier in states.values():
            for state in frontier:
                records = self._reconstruct_options(state)
                reclaim = int(state["reclaim"])
                if reclaim < target_reclaim_bytes:
                    continue
                miss_bytes = float(state["miss_bytes"])
                total_access_bytes = sum(
                    int(layer_bytes[layer]) / full_slots * self.top_k
                    for layer in layers
                )
                hit_rate = max(0.0, min(1.0, (
                    1.0 - miss_bytes / total_access_bytes
                    if total_access_bytes
                    else 1.0
                )))
                miss_gib = miss_bytes / 1024**3
                miss_gib_s = miss_gib * token_rate
                miss_latency_ms = float(state["miss_latency_ms"])
                shrink_copy = int(state["shrink_copy"])
                immediate_expand = int(state["immediate_expand"])
                future_expand = int(state["future_expand"])
                immediate_seconds = float(state["immediate_seconds"])
                future_seconds = float(state["future_seconds"])
                transition_seconds = float(state["transition_seconds"])
                amortized_ms = transition_seconds * 1000.0 / remaining_decode_tokens
                slowdown = (
                    baseline_tpot_ms + miss_latency_ms + amortized_ms
                ) / baseline_tpot_ms
                if hit_rate < self.minimum_byte_hit_rate:
                    continue
                if miss_gib_s > io_budget_gib_s:
                    continue
                if immediate_seconds * 1000.0 > release_deadline_ms:
                    continue
                if slowdown >= self.maximum_slowdown_ratio:
                    continue
                slots_by_layer = {
                    int(item["layer"]): int(item["slots"]) for item in records
                }
                decision = LayerResidencyDecision(
                    action="decode_elastic",
                    slots_per_layer=min(slots_by_layer.values(), default=full_slots),
                    slots_by_layer=slots_by_layer,
                    projected_byte_hit_rate=hit_rate,
                    miss_gib_per_token=miss_gib,
                    miss_gib_s=miss_gib_s,
                    estimated_slowdown_ratio=slowdown,
                    reason=(
                        "per-layer plan meets capacity, held-out route, I/O, "
                        "transition, and horizon guardrails using a "
                        "worst-observed source-cost surrogate"
                    ),
                    projected_reclaim_bytes=reclaim,
                    shrink_copy_bytes=shrink_copy,
                    immediate_expand_bytes=immediate_expand,
                    future_expand_bytes=future_expand,
                    immediate_transition_seconds=immediate_seconds,
                    future_transition_seconds=future_seconds,
                    transition_seconds=transition_seconds,
                    amortized_transition_ms_per_token=amortized_ms,
                    miss_latency_ms_per_token=miss_latency_ms,
                    remaining_decode_tokens=remaining_decode_tokens,
                    prediction_windows=(
                        int(state["minimum_windows"])
                        if math.isfinite(state["minimum_windows"])
                        else 0
                    ),
                    planner_wall_ms=(time.perf_counter() - planner_started) * 1000.0,
                    planner_peak_states=peak_states,
                )
                feasible.append((slowdown, reclaim - target_reclaim_bytes, decision))

        if not feasible:
            return self._fallback(
                "yield",
                full,
                remaining_decode_tokens,
                "no per-layer profile satisfies capacity, next-window risk, transition, and horizon bounds",
            )
        return min(feasible, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        minimized = (
            "objective_ms",
            "mean_misses",
            "miss_bytes",
            "miss_latency_ms",
            "immediate_seconds",
            "transition_seconds",
        )
        return int(left["reclaim"]) >= int(right["reclaim"]) and all(
            float(left[key]) <= float(right[key]) for key in minimized
        )

    @classmethod
    def _insert_frontier(
        cls, frontier: list[dict[str, Any]], candidate: dict[str, Any]
    ) -> None:
        survivors: list[dict[str, Any]] = []
        for existing in frontier:
            if cls._dominates(existing, candidate):
                return
            if not cls._dominates(candidate, existing):
                survivors.append(existing)
        frontier[:] = survivors
        frontier.append(candidate)

    @staticmethod
    def _reconstruct_options(state: dict[str, Any]) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        cursor: dict[str, Any] | None = state
        while cursor is not None and cursor.get("option") is not None:
            options.append(cursor["option"])
            cursor = cursor.get("parent")
        options.reverse()
        return options

    def _batch_latency_ms(self, misses: float) -> float:
        if misses <= 0:
            return 0.0
        points = sorted(self.miss_latency_curve_ms.items())
        if misses <= points[0][0]:
            return points[0][1] * misses / points[0][0]
        for (left_n, left_ms), (right_n, right_ms) in zip(points, points[1:]):
            if misses <= right_n:
                ratio = (misses - left_n) / (right_n - left_n)
                return left_ms + ratio * (right_ms - left_ms)
        largest_n, largest_ms = points[-1]
        full_batches, remainder = divmod(misses, largest_n)
        return full_batches * largest_ms + (
            self._batch_latency_ms(remainder) if remainder else 0.0
        )

    @staticmethod
    def _fallback(
        action: str,
        slots_by_layer: dict[int, int],
        remaining_decode_tokens: int,
        reason: str,
    ) -> LayerResidencyDecision:
        slots = min(slots_by_layer.values(), default=0)
        return LayerResidencyDecision(
            action=action,
            slots_per_layer=slots,
            slots_by_layer=slots_by_layer,
            projected_byte_hit_rate=1.0 if action == "full_resident" else 0.0,
            miss_gib_per_token=0.0,
            miss_gib_s=0.0,
            estimated_slowdown_ratio=1.0 if action == "full_resident" else math.inf,
            reason=reason,
            remaining_decode_tokens=remaining_decode_tokens,
            evidence="phase_or_feasibility_fallback",
        )


@dataclass(slots=True, frozen=True)
class DecodeCacheSimulation:
    policy: str
    slots_per_layer: int
    decode_tokens: int
    expert_accesses: int
    resident_hits: int
    blocking_misses: int
    byte_hit_rate: float
    miss_bytes: int
    misses_per_token_p50: float
    misses_per_token_p95: float
    protect_recent_tokens: int = 0
    exact_route_preserved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simulate_decode_cache(
    prefill_tail: Any,
    decode_routes: Any,
    slots_per_layer: int,
    expert_bytes: int,
    policy: str = "lru",
    history_window: int = 64,
    experts_per_layer: int | None = None,
    protect_recent_tokens: int = 0,
) -> DecodeCacheSimulation:
    """Replay real route arrays without executing or approximating experts."""
    if policy not in {"lru", "window_lfu"}:
        raise ValueError("policy must be lru or window_lfu")
    if (
        slots_per_layer <= 0
        or expert_bytes <= 0
        or history_window <= 0
        or protect_recent_tokens < 0
    ):
        raise ValueError(
            "slots, expert bytes, and history window must be positive; "
            "recent protection must be non-negative"
        )
    if getattr(decode_routes, "ndim", 0) != 3:
        raise ValueError("decode_routes must have shape [tokens, layers, top_k]")
    if getattr(prefill_tail, "ndim", 0) != 3:
        raise ValueError("prefill_tail must have shape [tokens, layers, top_k]")
    if prefill_tail.shape[1:] != decode_routes.shape[1:]:
        raise ValueError("prefill and decode route layouts differ")
    if experts_per_layer is None:
        maximum = max(
            int(prefill_tail.max()) if prefill_tail.size else -1,
            int(decode_routes.max()) if decode_routes.size else -1,
        )
        experts_per_layer = maximum + 1
    if experts_per_layer <= 0 or slots_per_layer > experts_per_layer:
        raise ValueError("slots must not exceed the model expert count")

    layer_count = int(decode_routes.shape[1])
    caches: list[dict[int, int]] = [dict() for _ in range(layer_count)]
    frequencies: list[Counter[int]] = [Counter() for _ in range(layer_count)]
    windows: list[deque[tuple[int, ...]]] = [
        deque(maxlen=history_window) for _ in range(layer_count)
    ]
    clock = 0

    for layer in range(layer_count):
        counts = Counter(int(item) for item in prefill_tail[:, layer, :].reshape(-1))
        recency: dict[int, int] = {}
        for token_index, row in enumerate(prefill_tail[:, layer, :]):
            for expert in row:
                recency[int(expert)] = token_index
        if slots_per_layer == experts_per_layer:
            initial = list(range(experts_per_layer))
        else:
            initial = sorted(
                range(experts_per_layer),
                key=lambda expert: (
                    -counts[expert],
                    -recency.get(expert, -1),
                    expert,
                ),
            )[:slots_per_layer]
        for expert in initial:
            clock += 1
            caches[layer][expert] = clock
        frequencies[layer].update(counts)

    accesses = 0
    hits = 0
    misses = 0
    misses_per_token: list[int] = []
    for token_routes in decode_routes:
        token_misses = 0
        for layer, row in enumerate(token_routes):
            actual = tuple(dict.fromkeys(int(expert) for expert in row))
            if len(actual) > slots_per_layer:
                raise ValueError("one exact Top-k route exceeds the physical slot count")
            cache = caches[layer]
            protected = set(actual)
            preferred = set(protected)
            if protect_recent_tokens:
                for recent_row in list(windows[layer])[-protect_recent_tokens:]:
                    preferred.update(recent_row)
            accesses += len(actual)
            for expert in actual:
                if expert in cache:
                    hits += 1
                else:
                    misses += 1
                    token_misses += 1
                    if len(cache) >= slots_per_layer:
                        candidates = [item for item in cache if item not in preferred]
                        if not candidates and preferred != protected:
                            candidates = [
                                item for item in cache if item not in protected
                            ]
                        if not candidates:
                            raise RuntimeError("no evictable expert remains for exact routing")
                        if policy == "window_lfu":
                            victim = min(
                                candidates,
                                key=lambda item: (
                                    frequencies[layer].get(item, 0),
                                    cache[item],
                                    item,
                                ),
                            )
                        else:
                            victim = min(candidates, key=lambda item: cache[item])
                        del cache[victim]
                clock += 1
                cache[expert] = clock

            if len(windows[layer]) == windows[layer].maxlen:
                frequencies[layer].subtract(windows[layer][0])
                frequencies[layer] += Counter()
            windows[layer].append(actual)
            frequencies[layer].update(actual)
        misses_per_token.append(token_misses)

    ordered = sorted(misses_per_token)

    def percentile(fraction: float) -> float:
        if not ordered:
            return 0.0
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
        return float(ordered[index])

    return DecodeCacheSimulation(
        policy=policy,
        slots_per_layer=slots_per_layer,
        decode_tokens=int(decode_routes.shape[0]),
        expert_accesses=accesses,
        resident_hits=hits,
        blocking_misses=misses,
        byte_hit_rate=hits / accesses if accesses else 0.0,
        miss_bytes=misses * int(expert_bytes),
        misses_per_token_p50=percentile(0.50),
        misses_per_token_p95=percentile(0.95),
        protect_recent_tokens=int(protect_recent_tokens),
    )
