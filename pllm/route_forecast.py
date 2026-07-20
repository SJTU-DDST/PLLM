from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class CoverageEstimate:
    slots_per_layer: int
    samples: int
    misses: int
    empirical_miss_rate: float
    upper_miss_rate: float
    certified: bool
    method: str = "one_sided_hoeffding_iid_assumption"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SequentialCoverageCalibrator:
    """Held-out all-layer route coverage gate.

    The Hoeffding bound is intentionally conservative and is reported with its
    IID assumption. It is a deployment gate, not a claim that decode routes are
    exchangeable.
    """

    def __init__(
        self,
        candidates: Iterable[int],
        *,
        target_miss_rate: float = 0.05,
        confidence_delta: float = 0.01,
        minimum_samples: int = 128,
    ) -> None:
        values = tuple(sorted({int(value) for value in candidates}))
        if not values or values[0] <= 0:
            raise ValueError("coverage candidates must be positive")
        if not 0.0 < target_miss_rate < 1.0:
            raise ValueError("target_miss_rate must be within (0, 1)")
        if not 0.0 < confidence_delta < 1.0:
            raise ValueError("confidence_delta must be within (0, 1)")
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        self.candidates = values
        self.target_miss_rate = float(target_miss_rate)
        self.confidence_delta = float(confidence_delta)
        self.minimum_samples = int(minimum_samples)
        self.samples = 0
        self.misses = Counter()

    def observe(self, missed_by_slots: Mapping[int, bool]) -> None:
        missing = set(self.candidates) - {int(key) for key in missed_by_slots}
        if missing:
            raise ValueError(f"coverage observation is missing candidates: {missing}")
        self.samples += 1
        for slots in self.candidates:
            if bool(missed_by_slots[slots]):
                self.misses[slots] += 1

    def estimate(self, slots: int) -> CoverageEstimate:
        slots = int(slots)
        if slots not in self.candidates:
            raise KeyError(f"slot profile {slots} is not calibrated")
        misses = int(self.misses[slots])
        empirical = misses / self.samples if self.samples else 1.0
        radius = (
            math.sqrt(math.log(1.0 / self.confidence_delta) / (2 * self.samples))
            if self.samples
            else 1.0
        )
        upper = min(1.0, empirical + radius)
        return CoverageEstimate(
            slots_per_layer=slots,
            samples=self.samples,
            misses=misses,
            empirical_miss_rate=empirical,
            upper_miss_rate=upper,
            certified=(
                self.samples >= self.minimum_samples
                and upper <= self.target_miss_rate
            ),
        )

    def status(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "target_miss_rate": self.target_miss_rate,
            "confidence_delta": self.confidence_delta,
            "minimum_samples": self.minimum_samples,
            "profiles": {
                str(slots): self.estimate(slots).to_dict()
                for slots in self.candidates
            },
            "scope": "any_exact_topk_miss_across_all_moe_layers_in_one_step",
            "assumption": "IID bound used as conservative gate; temporal drift monitored separately",
        }


@dataclass(slots=True, frozen=True)
class LayerForecast:
    layer: int
    slots_per_layer: int
    resident_experts: tuple[int, ...]
    core_bank: tuple[int, ...]
    forecast_bank: tuple[int, ...]
    emergency_bank: tuple[int, ...]
    mtp_signal_used: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "resident_experts",
            "core_bank",
            "forecast_bank",
            "emergency_bank",
        ):
            payload[key] = list(payload[key])
        return payload


class RouteMTPResidencyPredictor:
    """Exactness-preserving route forecast and residency planner.

    MTP can enter in two forms: its own Top-k experts, learned through a sparse
    cross-router transition table, or direct per-target-layer scores from future
    route heads. Neither signal is allowed to replace the target model's actual
    Top-k route.
    """

    def __init__(
        self,
        layers: Iterable[int],
        experts_per_layer: int,
        active_experts: int,
        *,
        candidate_slots: Iterable[int] = (256, 320, 384, 448, 480, 496, 504, 512),
        target_miss_rate: float = 0.05,
        confidence_delta: float = 0.01,
        minimum_calibration_samples: int = 128,
        transition_history: int = 32,
        transition_sources: int = 4,
        transition_width: int = 32,
    ) -> None:
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        if not self.layers:
            raise ValueError("at least one target MoE layer is required")
        if experts_per_layer <= 0 or active_experts <= 0:
            raise ValueError("expert counts must be positive")
        if active_experts > experts_per_layer:
            raise ValueError("active experts cannot exceed total experts")
        self.experts_per_layer = int(experts_per_layer)
        self.active_experts = int(active_experts)
        if transition_sources <= 0 or transition_width <= 0:
            raise ValueError("transition source and width limits must be positive")
        self.transition_sources = int(transition_sources)
        self.transition_width = int(transition_width)
        candidates = tuple(
            sorted(
                {
                    max(self.active_experts, min(int(value), self.experts_per_layer))
                    for value in candidate_slots
                }
            )
        )
        self.calibrator = SequentialCoverageCalibrator(
            candidates,
            target_miss_rate=target_miss_rate,
            confidence_delta=confidence_delta,
            minimum_samples=minimum_calibration_samples,
        )
        self.global_counts = {
            layer: np.zeros(self.experts_per_layer, dtype=np.uint64)
            for layer in self.layers
        }
        self.request_counts = {
            layer: np.zeros(self.experts_per_layer, dtype=np.uint64)
            for layer in self.layers
        }
        self.last_routes: dict[int, tuple[int, ...]] = {}
        self.temporal: dict[int, dict[int, Counter[int]]] = {
            layer: defaultdict(Counter) for layer in self.layers
        }
        self.mtp_cross_router: dict[int, dict[int, Counter[int]]] = {
            layer: defaultdict(Counter) for layer in self.layers
        }
        self.emergency_recency: dict[int, deque[int]] = {
            layer: deque(maxlen=max(self.active_experts, int(transition_history)))
            for layer in self.layers
        }
        self.observed_steps = 0
        self.predicted_steps = 0
        self.mtp_signal_steps = 0
        self.last_required_slots = self.experts_per_layer
        self.last_missed_by_slots = {
            slots: True for slots in self.calibrator.candidates
        }

    def reset_request(self) -> None:
        for layer in self.layers:
            self.request_counts[layer].fill(0)
            self.emergency_recency[layer].clear()
        self.last_routes.clear()

    def forecast(
        self,
        slots_per_layer: int,
        *,
        mtp_experts: Sequence[int] | None = None,
        direct_scores: Mapping[int, Sequence[float] | Mapping[int, float]] | None = None,
    ) -> dict[int, LayerForecast]:
        slots = int(slots_per_layer)
        if slots < self.active_experts or slots > self.experts_per_layer:
            raise ValueError(
                f"slots_per_layer must be within [{self.active_experts}, "
                f"{self.experts_per_layer}]"
            )
        normalized_mtp = self._normalize_experts(mtp_experts or ())
        result: dict[int, LayerForecast] = {}
        for layer in self.layers:
            layer_direct = direct_scores.get(layer) if direct_scores else None
            ranking = self._ranking(layer, normalized_mtp, layer_direct)
            result[layer] = self._partition(layer, slots, ranking, bool(normalized_mtp or layer_direct))
        return result

    def observe_step(
        self,
        actual_by_layer: Mapping[int, Iterable[int]],
        *,
        mtp_experts: Sequence[int] | None = None,
        direct_scores: Mapping[int, Sequence[float] | Mapping[int, float]] | None = None,
    ) -> dict[str, Any]:
        actual = {
            int(layer): self._normalize_experts(experts)
            for layer, experts in actual_by_layer.items()
        }
        if set(actual) != set(self.layers):
            missing = sorted(set(self.layers) - set(actual))
            extra = sorted(set(actual) - set(self.layers))
            raise ValueError(f"route step layer mismatch; missing={missing}, extra={extra}")
        if any(not experts for experts in actual.values()):
            raise ValueError("each target layer must contain an exact Top-k route")

        normalized_mtp = self._normalize_experts(mtp_experts or ())
        rankings = {
            layer: self._ranking(
                layer,
                normalized_mtp,
                direct_scores.get(layer) if direct_scores else None,
            )
            for layer in self.layers
        }
        ranks = {
            layer: {
                expert: index + 1
                for index, expert in enumerate(rankings[layer])
            }
            for layer in self.layers
        }
        required_by_layer = {
            layer: max(ranks[layer][expert] for expert in actual[layer])
            for layer in self.layers
        }
        missed_by_slots = {
            slots: any(
                required_by_layer[layer] > slots for layer in self.layers
            )
            for slots in self.calibrator.candidates
        }
        warm = self.observed_steps > 0
        if warm:
            self.calibrator.observe(missed_by_slots)
            self.predicted_steps += 1
            if normalized_mtp or direct_scores:
                self.mtp_signal_steps += 1

        max_required = self.active_experts
        for layer in self.layers:
            max_required = max(max_required, required_by_layer[layer])
            previous = self.last_routes.get(layer, ())
            if previous:
                for source in previous[: self.transition_sources]:
                    counter = self.temporal[layer][source]
                    counter.update(actual[layer])
                    self._prune_counter(counter)
            if normalized_mtp:
                for source in normalized_mtp[: self.transition_sources]:
                    counter = self.mtp_cross_router[layer][source]
                    counter.update(actual[layer])
                    self._prune_counter(counter)
            actual_indices = np.fromiter(actual[layer], dtype=np.int64)
            np.add.at(self.global_counts[layer], actual_indices, 1)
            np.add.at(self.request_counts[layer], actual_indices, 1)
            self.last_routes[layer] = actual[layer]

            resident = set(rankings[layer][: self.calibrator.candidates[0]])
            for expert in actual[layer]:
                if expert not in resident:
                    self._touch_emergency(layer, expert)

        self.observed_steps += 1
        self.last_required_slots = max_required
        self.last_missed_by_slots = missed_by_slots
        return {
            "warm_prediction": warm,
            "mtp_signal_used": bool(normalized_mtp or direct_scores),
            "required_uniform_rank": max_required,
            "missed_by_slots": {
                str(slots): missed for slots, missed in missed_by_slots.items()
            },
            "exact_route_authoritative": True,
        }

    def residency_plan(
        self,
        slots_per_layer: int,
        *,
        mtp_experts: Sequence[int] | None = None,
        direct_scores: Mapping[int, Sequence[float] | Mapping[int, float]] | None = None,
    ) -> dict[str, Any]:
        slots = int(slots_per_layer)
        forecasts = self.forecast(
            slots,
            mtp_experts=mtp_experts,
            direct_scores=direct_scores,
        )
        estimate = (
            self.calibrator.estimate(slots)
            if slots in self.calibrator.candidates
            else None
        )
        certified = bool(estimate and estimate.certified)
        return {
            "action": (
                "full_resident"
                if slots >= self.experts_per_layer
                else "prefetch_and_evict" if certified else "shadow_only"
            ),
            "slots_per_layer": slots,
            "layers": {
                str(layer): forecast.to_dict()
                for layer, forecast in forecasts.items()
            },
            "coverage": estimate.to_dict() if estimate else None,
            "exact_route_authoritative": True,
            "exact_miss_fallback_required": True,
            "reason": (
                "full profile retains every expert"
                if slots >= self.experts_per_layer
                else "held-out all-layer miss bound is certified"
                if certified
                else "insufficient or unsafe held-out route coverage"
            ),
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": "route_mtp_risk_calibrated_residency",
            "mode": "shadow",
            "observed_steps": self.observed_steps,
            "predicted_steps": self.predicted_steps,
            "mtp_signal_steps": self.mtp_signal_steps,
            "mtp_signal_attached": self.mtp_signal_steps > 0,
            "last_required_uniform_rank": self.last_required_slots,
            "last_missed_by_slots": {
                str(slots): missed
                for slots, missed in self.last_missed_by_slots.items()
            },
            "coverage": self.calibrator.status(),
            "exact_route_authoritative": True,
            "eviction_enabled": False,
            "evidence": "heldout_online_shadow_routes",
        }

    def _ranking(
        self,
        layer: int,
        mtp_experts: tuple[int, ...],
        direct_scores: Sequence[float] | Mapping[int, float] | None,
    ) -> list[int]:
        scores = np.zeros(self.experts_per_layer, dtype=np.float64)
        self._add_normalized(scores, self.global_counts[layer], 0.20)
        self._add_normalized(scores, self.request_counts[layer], 0.25)
        previous = self.last_routes.get(layer, ())
        if previous:
            temporal = Counter()
            for source in previous[: self.transition_sources]:
                temporal.update(self.temporal[layer].get(source, {}))
            self._add_normalized(scores, temporal, 0.20)
            previous_indices = np.fromiter(previous, dtype=np.int64)
            scores[previous_indices] += 0.05 / len(previous)
        if mtp_experts:
            mtp_counts = Counter()
            for source in mtp_experts[: self.transition_sources]:
                mtp_counts.update(self.mtp_cross_router[layer].get(source, {}))
            self._add_normalized(scores, mtp_counts, 0.20)
        if direct_scores is not None:
            if isinstance(direct_scores, Mapping):
                self._add_normalized(scores, direct_scores, 0.50)
            else:
                if len(direct_scores) != self.experts_per_layer:
                    raise ValueError(
                        f"layer {layer} direct MTP scores must contain "
                        f"{self.experts_per_layer} values"
                    )
                values = np.maximum(
                    np.asarray(direct_scores, dtype=np.float64), 0.0
                )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"layer {layer} direct MTP scores must be finite")
                total = float(values.sum())
                if total > 0.0:
                    scores += 0.50 * values / total
        # Stable descending argsort preserves expert ID order for exact ties.
        return np.argsort(-scores, kind="stable").tolist()

    def _partition(
        self,
        layer: int,
        slots: int,
        ranking: Sequence[int],
        mtp_signal_used: bool,
    ) -> LayerForecast:
        resident = tuple(ranking[:slots])
        resident_set = set(resident)
        core_limit = min(slots // 4, max(0, slots - self.active_experts))
        emergency_limit = min(max(self.active_experts, slots // 16), slots)
        core_ranking = np.argsort(
            -self.global_counts[layer].astype(np.float64), kind="stable"
        ).tolist()
        core = tuple(
            expert for expert in core_ranking if expert in resident_set
        )[:core_limit]
        classified = set(core)
        emergency_values: list[int] = []
        for expert in reversed(self.emergency_recency[layer]):
            if (
                expert not in resident_set
                or expert in classified
                or expert in emergency_values
            ):
                continue
            emergency_values.append(expert)
            if len(emergency_values) >= emergency_limit:
                break
        classified.update(emergency_values)
        forecast_values = [expert for expert in resident if expert not in classified]
        if len(resident) != slots or len(resident_set) != slots:
            raise RuntimeError("route forecast did not produce a complete unique residency set")
        return LayerForecast(
            layer=layer,
            slots_per_layer=slots,
            resident_experts=resident,
            core_bank=core,
            forecast_bank=tuple(forecast_values),
            emergency_bank=tuple(emergency_values),
            mtp_signal_used=mtp_signal_used,
        )

    def _touch_emergency(self, layer: int, expert: int) -> None:
        values = self.emergency_recency[layer]
        try:
            values.remove(expert)
        except ValueError:
            pass
        values.append(expert)

    def _prune_counter(self, counter: Counter[int]) -> None:
        if len(counter) <= self.transition_width:
            return
        retained = counter.most_common(self.transition_width)
        counter.clear()
        counter.update(dict(retained))

    def _normalize_experts(self, experts: Iterable[int]) -> tuple[int, ...]:
        values = tuple(dict.fromkeys(int(expert) for expert in experts))
        if any(expert < 0 or expert >= self.experts_per_layer for expert in values):
            raise ValueError("route contains an out-of-range expert")
        return values

    @staticmethod
    def _add_normalized(
        target: np.ndarray,
        values: Mapping[int, int | float] | np.ndarray,
        weight: float,
    ) -> None:
        if isinstance(values, np.ndarray):
            if values.size != target.size:
                raise ValueError("route count vector size does not match expert count")
            positive = np.maximum(values.astype(np.float64, copy=False), 0.0)
            total = float(positive.sum())
            if total > 0.0:
                target += weight * positive / total
            return
        if not values:
            return
        indices = np.fromiter(
            (int(index) for index in values.keys()),
            dtype=np.int64,
            count=len(values),
        )
        raw = np.fromiter(
            (float(value) for value in values.values()),
            dtype=np.float64,
            count=len(values),
        )
        valid = (indices >= 0) & (indices < target.size) & (raw > 0.0)
        if not np.all(np.isfinite(raw[valid])):
            raise ValueError("route forecast scores must be finite")
        indices = indices[valid]
        positive = raw[valid]
        total = float(positive.sum())
        if total <= 0.0:
            return
        target[indices] += weight * positive / total
