from __future__ import annotations

import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .expert_catalog import GIB, ExpertCatalog
from .expert_trace import ExpertRouteRecord


class RouteHistoryPredictor:
    """A lightweight trace predictor used by the no-GPU control-plane prototype."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self.popularity: dict[int, Counter[int]] = defaultdict(Counter)
        self.transitions: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)

    def fit(self, records: Iterable[ExpertRouteRecord]) -> None:
        previous: dict[tuple[str, int], list[int]] = {}
        for record in records:
            self.popularity[record.layer].update(record.actual_experts)
            key = (record.request_id, record.layer)
            for prior in previous.get(key, []):
                self.transitions[(record.layer, prior)].update(record.actual_experts)
            previous[key] = list(record.actual_experts)

    def rank(self, layer: int, recent_experts: list[int] | None = None) -> list[int]:
        scores = Counter(self.popularity.get(layer, {}))
        for expert in recent_experts or []:
            transition = self.transitions.get((layer, expert))
            if transition:
                for candidate, count in transition.items():
                    scores[candidate] += count * 2
            scores[expert] += 1
        return sorted(
            range(self.num_experts),
            key=lambda expert: (-scores[expert], expert),
        )


@dataclass(slots=True)
class CalibrationReport:
    alpha: float
    records: int
    thresholds_by_layer: dict[int, int]
    empirical_coverage: float
    average_set_size: float
    guarantee_scope: str = "split_conformal_marginal_under_exchangeability"
    evidence: str = "trace_dependent"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConformalExpertPredictor:
    def __init__(self, predictor: RouteHistoryPredictor, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.predictor = predictor
        self.alpha = alpha
        self.thresholds_by_layer: dict[int, int] = {}

    def calibrate(self, records: Iterable[ExpertRouteRecord]) -> CalibrationReport:
        rows = list(records)
        required_ranks: dict[int, list[int]] = defaultdict(list)
        previous: dict[tuple[str, int], list[int]] = {}
        for record in rows:
            key = (record.request_id, record.layer)
            ranking = self.predictor.rank(record.layer, previous.get(key))
            positions = {expert: index + 1 for index, expert in enumerate(ranking)}
            required_ranks[record.layer].append(
                max(positions[expert] for expert in record.actual_experts)
            )
            previous[key] = list(record.actual_experts)

        for layer, values in required_ranks.items():
            ordered = sorted(values)
            quantile_index = min(
                len(ordered) - 1,
                max(0, math.ceil((len(ordered) + 1) * (1 - self.alpha)) - 1),
            )
            self.thresholds_by_layer[layer] = ordered[quantile_index]

        evaluated = self.evaluate(rows)
        return CalibrationReport(
            alpha=self.alpha,
            records=len(rows),
            thresholds_by_layer=dict(self.thresholds_by_layer),
            empirical_coverage=evaluated["coverage"],
            average_set_size=evaluated["average_set_size"],
        )

    def prediction_set(
        self, layer: int, recent_experts: list[int] | None = None
    ) -> list[int]:
        if layer not in self.thresholds_by_layer:
            raise ValueError(f"layer {layer} has not been calibrated")
        ranking = self.predictor.rank(layer, recent_experts)
        return ranking[: self.thresholds_by_layer[layer]]

    def evaluate(self, records: Iterable[ExpertRouteRecord]) -> dict[str, float]:
        previous: dict[tuple[str, int], list[int]] = {}
        covered = 0
        total = 0
        set_sizes = 0
        for record in records:
            key = (record.request_id, record.layer)
            predicted = set(self.prediction_set(record.layer, previous.get(key)))
            covered += set(record.actual_experts).issubset(predicted)
            total += 1
            set_sizes += len(predicted)
            previous[key] = list(record.actual_experts)
        return {
            "coverage": covered / total if total else 0.0,
            "average_set_size": set_sizes / total if total else 0.0,
        }


@dataclass(slots=True)
class CacheSimulationResult:
    records: int
    slots_per_layer: int
    actual_bytes: int
    resident_hit_bytes: int
    useful_prefetch_bytes: int
    blocking_miss_bytes: int
    false_prefetch_bytes: int
    evicted_bytes: int
    prediction_over_budget_records: int
    exact_route_preserved: bool
    evidence: str

    @property
    def byte_hit_rate(self) -> float:
        ready = self.resident_hit_bytes + self.useful_prefetch_bytes
        return ready / self.actual_bytes if self.actual_bytes else 0.0

    @property
    def resident_hit_rate(self) -> float:
        return self.resident_hit_bytes / self.actual_bytes if self.actual_bytes else 0.0

    @property
    def blocking_miss_bytes_per_record(self) -> float:
        return self.blocking_miss_bytes / self.records if self.records else 0.0

    @property
    def false_prefetch_bytes_per_record(self) -> float:
        return self.false_prefetch_bytes / self.records if self.records else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "byte_hit_rate": round(self.byte_hit_rate, 6),
                "resident_hit_rate": round(self.resident_hit_rate, 6),
                "blocking_miss_bytes_per_record": round(
                    self.blocking_miss_bytes_per_record, 3
                ),
                "false_prefetch_bytes_per_record": round(
                    self.false_prefetch_bytes_per_record, 3
                ),
            }
        )
        return payload


class ExpertCacheSimulator:
    def __init__(
        self,
        catalog: ExpertCatalog,
        predictor: ConformalExpertPredictor,
        slots_per_layer: int,
        evidence: str = "synthetic_no_gpu",
    ) -> None:
        if slots_per_layer < catalog.active_experts_per_token:
            raise ValueError("slots_per_layer cannot be smaller than the model Top-k")
        if slots_per_layer > catalog.experts_per_layer:
            raise ValueError("slots_per_layer exceeds the experts per layer")
        self.catalog = catalog
        self.predictor = predictor
        self.slots_per_layer = slots_per_layer
        self.evidence = evidence
        self._sizes = {
            (item.layer, item.expert): item.size_bytes for item in catalog.experts
        }

    def run(self, records: Iterable[ExpertRouteRecord]) -> CacheSimulationResult:
        caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        previous: dict[tuple[str, int], list[int]] = {}
        metrics = {
            "records": 0,
            "actual_bytes": 0,
            "resident_hit_bytes": 0,
            "useful_prefetch_bytes": 0,
            "blocking_miss_bytes": 0,
            "false_prefetch_bytes": 0,
            "evicted_bytes": 0,
            "prediction_over_budget_records": 0,
        }
        for record in records:
            metrics["records"] += 1
            key = (record.request_id, record.layer)
            cache = caches[record.layer]
            actual = set(record.actual_experts)
            predicted = self.predictor.prediction_set(
                record.layer, previous.get(key)
            )
            if len(predicted) > self.slots_per_layer:
                metrics["prediction_over_budget_records"] += 1
            scheduled = predicted[: self.slots_per_layer]
            resident_before = set(cache)
            newly_prefetched: set[int] = set()

            for expert in scheduled:
                if expert in cache:
                    cache.move_to_end(expert)
                    continue
                metrics["evicted_bytes"] += self._ensure_room(cache, record.layer)
                cache[expert] = None
                newly_prefetched.add(expert)

            for expert in newly_prefetched:
                if expert not in actual or expert not in cache:
                    metrics["false_prefetch_bytes"] += self._size(
                        record.layer, expert
                    )

            for expert in record.actual_experts:
                size = self._size(record.layer, expert)
                metrics["actual_bytes"] += size
                if expert in cache:
                    if expert in resident_before:
                        metrics["resident_hit_bytes"] += size
                    elif expert in newly_prefetched:
                        metrics["useful_prefetch_bytes"] += size
                else:
                    metrics["evicted_bytes"] += self._ensure_room(cache, record.layer)
                    cache[expert] = None
                    metrics["blocking_miss_bytes"] += size
                cache.move_to_end(expert)
            previous[key] = list(record.actual_experts)

        return CacheSimulationResult(
            records=metrics["records"],
            slots_per_layer=self.slots_per_layer,
            actual_bytes=metrics["actual_bytes"],
            resident_hit_bytes=metrics["resident_hit_bytes"],
            useful_prefetch_bytes=metrics["useful_prefetch_bytes"],
            blocking_miss_bytes=metrics["blocking_miss_bytes"],
            false_prefetch_bytes=metrics["false_prefetch_bytes"],
            evicted_bytes=metrics["evicted_bytes"],
            prediction_over_budget_records=metrics[
                "prediction_over_budget_records"
            ],
            exact_route_preserved=True,
            evidence=self.evidence,
        )

    def _ensure_room(self, cache: OrderedDict[int, None], layer: int) -> int:
        if len(cache) < self.slots_per_layer:
            return 0
        evicted, _ = cache.popitem(last=False)
        return self._size(layer, evicted)

    def _size(self, layer: int, expert: int) -> int:
        return self._sizes.get((layer, expert), int(self.catalog.average_expert_bytes))


@dataclass(slots=True)
class ResourceEnvelope:
    total_memory_gib: float = 128.0
    foreground_reserve_gib: float = 32.0
    system_reserve_gib: float = 16.0
    io_budget_gib_s: float = 2.0
    compute_duty_cycle: float = 0.5
    requested_token_rate: float = 5.0
    minimum_token_rate: float = 0.5
    release_deadline_ms: float = 500.0


@dataclass(slots=True)
class ResidencyPlan:
    action: str
    slots_per_layer: int
    projected_resident_gib: float
    projected_reclaim_gib: float
    estimated_miss_gib_per_token: float
    estimated_miss_gib_s: float
    token_rate_limit: float
    reason: str
    exact_route_required: bool = True
    data_plane_ready: bool = False
    evidence: str = "control_plane_projection"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidencyPlanner:
    def __init__(
        self,
        catalog: ExpertCatalog,
        slot_profiles: tuple[int, ...] = (32, 64, 128, 256, 512),
    ) -> None:
        self.catalog = catalog
        self.slot_profiles = tuple(
            sorted(
                {
                    value
                    for value in slot_profiles
                    if catalog.active_experts_per_token
                    <= value
                    <= catalog.experts_per_layer
                }
            )
        )

    def plan(
        self,
        envelope: ResourceEnvelope,
        byte_hit_rate: float,
        false_prefetch_bytes_per_token: float = 0.0,
    ) -> ResidencyPlan:
        if not 0 <= byte_hit_rate <= 1:
            raise ValueError("byte_hit_rate must be in [0, 1]")
        if not 0 <= envelope.compute_duty_cycle <= 1:
            raise ValueError("compute_duty_cycle must be in [0, 1]")
        available_gib = max(
            0.0,
            envelope.total_memory_gib
            - envelope.foreground_reserve_gib
            - envelope.system_reserve_gib,
        )
        non_routed_gib = self.catalog.non_routed_bytes / GIB
        max_slots = math.floor(
            max(0.0, available_gib - non_routed_gib)
            * GIB
            * self.catalog.experts_per_layer
            / max(1, self.catalog.routed_expert_bytes)
        )
        feasible_profiles = [value for value in self.slot_profiles if value <= max_slots]
        if not feasible_profiles:
            return self._hibernate(
                "foreground capacity leaves no viable exact Top-k expert working set"
            )
        slots = max(feasible_profiles)
        projection = self.catalog.project_slots(slots)
        miss_bytes_per_token = (
            self.catalog.active_expert_bytes_per_token * (1 - byte_hit_rate)
            + max(0.0, false_prefetch_bytes_per_token)
        )
        miss_gib_per_token = miss_bytes_per_token / GIB
        compute_rate = max(
            0.0, envelope.requested_token_rate * envelope.compute_duty_cycle
        )
        io_rate = (
            envelope.io_budget_gib_s / miss_gib_per_token
            if miss_gib_per_token > 0
            else envelope.requested_token_rate
        )
        rate_limit = min(envelope.requested_token_rate, compute_rate, io_rate)

        if envelope.compute_duty_cycle <= 0.05 and slots == self.catalog.experts_per_layer:
            return ResidencyPlan(
                action="yield",
                slots_per_layer=slots,
                projected_resident_gib=projection["resident_weight_gib"],
                projected_reclaim_gib=projection["projected_reclaim_gib"],
                estimated_miss_gib_per_token=round(miss_gib_per_token, 6),
                estimated_miss_gib_s=0.0,
                token_rate_limit=0.0,
                reason="capacity fits, but foreground compute envelope requires a token-boundary yield",
            )
        if rate_limit < envelope.minimum_token_rate:
            return self._hibernate(
                "predicted expert I/O or compute share cannot sustain the minimum token rate",
                miss_gib_per_token,
            )
        action = (
            "full_resident"
            if slots == self.catalog.experts_per_layer
            else "elastic_resident"
        )
        return ResidencyPlan(
            action=action,
            slots_per_layer=slots,
            projected_resident_gib=projection["resident_weight_gib"],
            projected_reclaim_gib=projection["projected_reclaim_gib"],
            estimated_miss_gib_per_token=round(miss_gib_per_token, 6),
            estimated_miss_gib_s=round(miss_gib_per_token * rate_limit, 6),
            token_rate_limit=round(rate_limit, 3),
            reason=(
                "full weights fit inside the foreground resource envelope"
                if action == "full_resident"
                else "exact expert residency is feasible under capacity and I/O budgets"
            ),
        )

    def _hibernate(
        self, reason: str, miss_gib_per_token: float = 0.0
    ) -> ResidencyPlan:
        return ResidencyPlan(
            action="hibernate",
            slots_per_layer=0,
            projected_resident_gib=0.0,
            projected_reclaim_gib=round(self.catalog.total_tensor_bytes / GIB, 3),
            estimated_miss_gib_per_token=round(miss_gib_per_token, 6),
            estimated_miss_gib_s=0.0,
            token_rate_limit=0.0,
            reason=reason,
        )
