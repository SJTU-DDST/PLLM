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


class DecodeRouteWindow:
    """Small CPU route window used for low-frequency decode residency changes."""

    def __init__(
        self,
        layers: Iterable[int],
        experts_per_layer: int,
        window_steps: int = 256,
    ) -> None:
        if experts_per_layer <= 0 or window_steps <= 0:
            raise ValueError("experts_per_layer and window_steps must be positive")
        self.layers = tuple(sorted(int(layer) for layer in layers))
        if not self.layers:
            raise ValueError("at least one MoE layer is required")
        self.experts_per_layer = int(experts_per_layer)
        self.window_steps = int(window_steps)
        self.phase = "idle"
        self.phase_changed_at = time.time()
        self._decode: dict[int, deque[tuple[int, ...]]] = {
            layer: deque(maxlen=self.window_steps) for layer in self.layers
        }
        self._prefill_tail: dict[int, deque[tuple[int, ...]]] = {
            layer: deque(maxlen=min(16, self.window_steps)) for layer in self.layers
        }
        self._decode_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self._prefill_counts: dict[int, Counter[int]] = defaultdict(Counter)
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
                for rows in self._decode.values():
                    rows.clear()
                self._decode_counts.clear()
                self._observations["decode"] = 0
                self._tokens["decode"] = 0

    def observe(
        self,
        layer: int,
        experts: Iterable[int],
        token_count: int = 1,
    ) -> None:
        layer = int(layer)
        unique = tuple(dict.fromkeys(int(expert) for expert in experts))
        if layer not in self._decode:
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
                rows = self._decode[layer]
                if len(rows) == rows.maxlen:
                    self._decode_counts[layer].subtract(rows[0])
                    self._decode_counts[layer] += Counter()
                rows.append(unique)
                self._decode_counts[layer].update(unique)
            elif phase == "prefill":
                rows = self._prefill_tail[layer]
                if len(rows) == rows.maxlen:
                    self._prefill_counts[layer].subtract(rows[0])
                    self._prefill_counts[layer] += Counter()
                rows.append(unique)
                self._prefill_counts[layer].update(unique)

    def hot_experts(self, layer: int, limit: int) -> list[int]:
        if limit <= 0:
            return []
        with self._lock:
            decode_counts = self._decode_counts.get(layer, Counter())
            prefill_counts = self._prefill_counts.get(layer, Counter())
            recent: dict[int, int] = {}
            sequence = list(self._prefill_tail.get(layer, ())) + list(
                self._decode.get(layer, ())
            )
            for position, row in enumerate(sequence, start=1):
                for expert in row:
                    recent[expert] = position
            ranking = sorted(
                range(self.experts_per_layer),
                key=lambda expert: (
                    -decode_counts.get(expert, 0),
                    -prefill_counts.get(expert, 0),
                    -recent.get(expert, 0),
                    expert,
                ),
            )
            return ranking[: min(int(limit), self.experts_per_layer)]

    def projected_hit_rate(self, slots_per_layer: int) -> float:
        slots = max(0, min(int(slots_per_layer), self.experts_per_layer))
        if slots >= self.experts_per_layer:
            return 1.0
        with self._lock:
            hits = 0
            accesses = 0
            for layer in self.layers:
                counts = self._decode_counts.get(layer, Counter())
                if not counts:
                    continue
                hot = set(self.hot_experts(layer, slots))
                hits += sum(count for expert, count in counts.items() if expert in hot)
                accesses += sum(counts.values())
            return hits / accesses if accesses else 0.0

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
            decode_rows = sum(len(rows) for rows in self._decode.values())
            prefill_rows = sum(len(rows) for rows in self._prefill_tail.values())
            projected = {
                str(profile): round(self.projected_hit_rate(profile), 6)
                for profile in profiles
                if 0 < int(profile) <= self.experts_per_layer
            }
            return {
                "phase": self.phase,
                "phase_changed_at": self.phase_changed_at,
                "window_steps": self.window_steps,
                "decode_observations": int(self._observations["decode"]),
                "decode_token_rows": int(self._tokens["decode"]),
                "decode_layer_rows_retained": decode_rows,
                "prefill_layer_rows_retained": prefill_rows,
                "projected_byte_hit_rate": projected,
                "storage": "cpu_ring",
                "gpu_persistent_bytes": 0,
                "evidence": "runtime_actual_topk_union_per_layer_step",
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
) -> DecodeCacheSimulation:
    """Replay real route arrays without executing or approximating experts."""
    if policy not in {"lru", "window_lfu"}:
        raise ValueError("policy must be lru or window_lfu")
    if slots_per_layer <= 0 or expert_bytes <= 0 or history_window <= 0:
        raise ValueError("slots, expert bytes, and history window must be positive")
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
            accesses += len(actual)
            for expert in actual:
                if expert in cache:
                    hits += 1
                else:
                    misses += 1
                    token_misses += 1
                    if len(cache) >= slots_per_layer:
                        candidates = [item for item in cache if item not in protected]
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
    )
