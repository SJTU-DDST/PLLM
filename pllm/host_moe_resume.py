from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence


RouteRow = Sequence[int]
RouteHistory = Sequence[Sequence[RouteRow]]


@dataclass(slots=True, frozen=True)
class HostMoeResumePlan:
    physical_slots: int
    hot_slots: int
    experts_per_layer: int
    hot_experts_by_layer: tuple[tuple[int, ...], ...]
    exact_misses_by_layer: tuple[tuple[int, ...], ...]
    next_routes_by_layer: tuple[tuple[int, ...], ...]

    @property
    def layers(self) -> int:
        return len(self.hot_experts_by_layer)

    @property
    def naive_copy_objects(self) -> int:
        return self.physical_slots * self.layers

    @property
    def exact_miss_objects(self) -> int:
        return sum(len(items) for items in self.exact_misses_by_layer)

    @property
    def critical_copy_objects(self) -> int:
        return self.hot_slots * self.layers + self.exact_miss_objects

    @property
    def expert_copy_reduction_ratio(self) -> float:
        if not self.naive_copy_objects:
            return 0.0
        return 1.0 - self.critical_copy_objects / self.naive_copy_objects

    @property
    def exact_route_covered(self) -> bool:
        return all(
            set(route).issubset(set(hot) | set(misses))
            for hot, misses, route in zip(
                self.hot_experts_by_layer,
                self.exact_misses_by_layer,
                self.next_routes_by_layer,
            )
        )


def rank_recent_experts(
    rows: Sequence[RouteRow], experts_per_layer: int
) -> tuple[int, ...]:
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")
    counts: Counter[int] = Counter()
    recency: dict[int, int] = {}
    for position, row in enumerate(rows):
        unique = tuple(dict.fromkeys(int(expert) for expert in row))
        if any(expert < 0 or expert >= experts_per_layer for expert in unique):
            raise ValueError("route history contains an out-of-range expert")
        counts.update(unique)
        for expert in unique:
            recency[expert] = position
    return tuple(
        sorted(
            range(experts_per_layer),
            key=lambda expert: (
                -counts[expert],
                -recency.get(expert, -1),
                expert,
            ),
        )
    )


def plan_host_moe_resume(
    history: RouteHistory,
    next_routes: Sequence[RouteRow],
    *,
    physical_slots: int,
    hot_slots: int,
    experts_per_layer: int,
) -> HostMoeResumePlan:
    if not 0 < hot_slots <= physical_slots <= experts_per_layer:
        raise ValueError(
            "resume slots must satisfy 0 < hot_slots <= physical_slots "
            "<= experts_per_layer"
        )
    layers = len(next_routes)
    if layers <= 0:
        raise ValueError("next_routes cannot be empty")
    if any(len(token) != layers for token in history):
        raise ValueError("every history token must contain every MoE layer")

    hot_by_layer: list[tuple[int, ...]] = []
    misses_by_layer: list[tuple[int, ...]] = []
    normalized_routes: list[tuple[int, ...]] = []
    for layer, route in enumerate(next_routes):
        exact = tuple(dict.fromkeys(int(expert) for expert in route))
        if any(expert < 0 or expert >= experts_per_layer for expert in exact):
            raise ValueError("next route contains an out-of-range expert")
        if len(exact) > physical_slots:
            raise ValueError("next route cannot fit in the physical expert slots")
        layer_history = [token[layer] for token in history]
        hot = rank_recent_experts(layer_history, experts_per_layer)[:hot_slots]
        hot_set = set(hot)
        misses = tuple(expert for expert in exact if expert not in hot_set)
        hot_by_layer.append(hot)
        misses_by_layer.append(misses)
        normalized_routes.append(exact)

    return HostMoeResumePlan(
        physical_slots=physical_slots,
        hot_slots=hot_slots,
        experts_per_layer=experts_per_layer,
        hot_experts_by_layer=tuple(hot_by_layer),
        exact_misses_by_layer=tuple(misses_by_layer),
        next_routes_by_layer=tuple(normalized_routes),
    )
