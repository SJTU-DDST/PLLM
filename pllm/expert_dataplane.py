from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .expert_store import ExpertPayload, ExpertSource


class SlotState(StrEnum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class SlotSink(Protocol):
    slot_count: int
    required_format: str

    def write(self, slot: int, payload: ExpertPayload) -> None: ...

    def invalidate(self, slot: int) -> None: ...

    def publish_mapping(self, mapping: dict[int, int], generation: int) -> None: ...

    def begin_resize(self, slot_count: int) -> None: ...

    def finish_resize(self) -> None: ...


@dataclass(slots=True)
class SlotRecord:
    slot: int
    state: SlotState = SlotState.EMPTY
    logical_expert: int | None = None
    generation: int = 0
    last_used_at: float = 0.0
    loads: int = 0
    error: str = ""


@dataclass(slots=True)
class LayerCounters:
    hits: int = 0
    misses: int = 0
    prefetches: int = 0
    evictions: int = 0
    bytes_loaded: int = 0
    load_time_ns: int = 0
    resize_count: int = 0
    resize_time_ns: int = 0
    gpu_copy_bytes: int = 0
    batch_loads: int = 0
    batch_objects: int = 0
    capacity_change_count: int = 0
    capacity_change_time_ns: int = 0


@dataclass(slots=True)
class LayerSlotState:
    layer: int
    global_experts: int
    sink: SlotSink
    slots: list[SlotRecord]
    active_slots: set[int]
    logical_to_slot: dict[int, int] = field(default_factory=dict)
    generation: int = 0
    counters: LayerCounters = field(default_factory=LayerCounters)
    lock: threading.RLock = field(default_factory=threading.RLock)


class ExpertSlotDataPlane:
    """Exact expert working-set manager shared by SSD and RDMA sources."""

    def __init__(self, source: ExpertSource) -> None:
        self.source = source
        self._layers: dict[int, LayerSlotState] = {}
        self._lock = threading.RLock()

    def register_layer(
        self,
        layer: int,
        global_experts: int,
        sink: SlotSink,
        initial_experts: list[int] | None = None,
    ) -> None:
        if sink.slot_count <= 0 or sink.slot_count > global_experts:
            raise ValueError("slot count must be within the logical expert range")
        with self._lock:
            if layer in self._layers:
                raise ValueError(f"layer {layer} is already registered")
            state = LayerSlotState(
                layer=layer,
                global_experts=global_experts,
                sink=sink,
                slots=[SlotRecord(slot=index) for index in range(sink.slot_count)],
                active_slots=set(range(sink.slot_count)),
            )
            self._layers[layer] = state
        if initial_experts:
            self.ensure(layer, initial_experts, reason="initial")
        else:
            sink.publish_mapping({}, 0)

    def unregister_layer(self, layer: int) -> None:
        with self._lock:
            state = self._layers.pop(layer, None)
        if state is None:
            return
        with state.lock:
            state.logical_to_slot.clear()
            state.generation += 1
            state.sink.publish_mapping({}, state.generation)

    def ensure(
        self,
        layer: int,
        experts: list[int] | tuple[int, ...] | set[int],
        reason: str = "route_miss",
        pinned_experts: list[int] | tuple[int, ...] | set[int] = (),
    ) -> dict[int, int]:
        state = self._layer(layer)
        requested = list(dict.fromkeys(int(item) for item in experts))
        if any(item < 0 or item >= state.global_experts for item in requested):
            raise ValueError(f"layer {layer} received an out-of-range expert ID")
        if len(requested) > len(state.active_slots):
            raise RuntimeError(
                f"layer {layer} needs {len(requested)} experts but has "
                f"only {len(state.active_slots)} active slots"
            )

        now = time.monotonic()
        with state.lock:
            missing: list[int] = []
            for expert in requested:
                slot_index = state.logical_to_slot.get(expert)
                if slot_index is None:
                    missing.append(expert)
                    continue
                slot = state.slots[slot_index]
                if slot.state != SlotState.READY:
                    missing.append(expert)
                    continue
                slot.last_used_at = now
                state.counters.hits += 1

            protected = set(requested)
            preferred = protected | {
                int(expert)
                for expert in pinned_experts
                if int(expert) in state.logical_to_slot
            }
            reservations: list[tuple[int, SlotRecord]] = []
            for expert in missing:
                state.counters.misses += 1
                if reason == "prefetch":
                    state.counters.prefetches += 1
                slot = self._select_slot(state, protected, preferred)
                self._remove_mapping(state, slot)
                slot.state = SlotState.LOADING
                slot.logical_expert = expert
                slot.error = ""
                state.generation += 1
                slot.generation = state.generation
                reservations.append((expert, slot))

            if reservations:
                state.sink.publish_mapping(
                    dict(state.logical_to_slot), state.generation
                )
                started = time.perf_counter_ns()
                loaded_bytes = 0
                loaded_objects = 0
                load_batches = 0
                try:
                    requests = [
                        (layer, expert) for expert, _slot in reservations
                    ]
                    iter_many = getattr(self.source, "iter_many", None)
                    if callable(iter_many):
                        payload_batches = iter_many(requests)
                    else:
                        get_many = getattr(self.source, "get_many", None)
                        payload_batches = [
                            get_many(requests)
                            if callable(get_many)
                            else [self.source.get(*request) for request in requests]
                        ]
                    for payloads in payload_batches:
                        selected = reservations[
                            loaded_objects : loaded_objects + len(payloads)
                        ]
                        if len(selected) != len(payloads):
                            raise RuntimeError("expert source returned an oversized batch")
                        for (expert, slot), payload in zip(selected, payloads):
                            if (payload.layer, payload.expert) != (layer, expert):
                                raise ValueError(
                                    "batched expert source changed request order"
                                )
                            if payload.format != state.sink.required_format:
                                raise ValueError(
                                    f"slot sink requires {state.sink.required_format}, "
                                    f"found {payload.format}"
                                )
                            state.sink.write(slot.slot, payload)
                            loaded_bytes += len(payload.data)
                        loaded_objects += len(payloads)
                        load_batches += 1
                    if loaded_objects != len(reservations):
                        raise RuntimeError("expert source returned a short batch")
                except Exception as exc:
                    for _expert, slot in reservations:
                        slot.state = SlotState.FAILED
                        slot.error = str(exc)
                        slot.logical_expert = None
                        state.sink.invalidate(slot.slot)
                    state.sink.publish_mapping(
                        dict(state.logical_to_slot), state.generation
                    )
                    raise
                elapsed = time.perf_counter_ns() - started
                for expert, slot in reservations:
                    slot.state = SlotState.READY
                    slot.last_used_at = time.monotonic()
                    slot.loads += 1
                    state.logical_to_slot[expert] = slot.slot
                state.counters.bytes_loaded += loaded_bytes
                state.counters.load_time_ns += elapsed
                state.counters.batch_loads += load_batches
                state.counters.batch_objects += len(reservations)

            if reservations:
                state.generation += 1
                state.sink.publish_mapping(
                    dict(state.logical_to_slot), state.generation
                )
            return {expert: state.logical_to_slot[expert] for expert in requested}

    def prefetch(self, layer: int, experts: list[int]) -> dict[int, int]:
        return self.ensure(layer, experts, reason="prefetch")

    def evict(self, layer: int, experts: list[int]) -> int:
        state = self._layer(layer)
        evicted = 0
        with state.lock:
            for expert in dict.fromkeys(int(item) for item in experts):
                slot_index = state.logical_to_slot.pop(expert, None)
                if slot_index is None:
                    continue
                slot = state.slots[slot_index]
                slot.state = SlotState.EMPTY
                slot.logical_expert = None
                slot.error = ""
                state.sink.invalidate(slot.slot)
                state.counters.evictions += 1
                evicted += 1
            if evicted:
                state.generation += 1
                state.sink.publish_mapping(
                    dict(state.logical_to_slot), state.generation
                )
        return evicted

    def evict_all(self) -> int:
        count = 0
        for layer in self.layers():
            state = self._layer(layer)
            count += self.evict(layer, list(state.logical_to_slot))
        return count

    def set_capacity(
        self, layer: int, capacity: int, retain: list[int] | None = None
    ) -> dict[str, Any]:
        """Limit cache residency without reallocating model weight tensors."""
        state = self._layer(layer)
        if capacity <= 0 or capacity > len(state.slots):
            raise ValueError("capacity must be within the physical slot range")
        with state.lock:
            started = time.perf_counter_ns()
            requested = list(dict.fromkeys(retain or []))
            ranked = [
                expert
                for expert in requested
                if expert in state.logical_to_slot
                and state.slots[state.logical_to_slot[expert]].state == SlotState.READY
            ]
            ranked.extend(
                expert
                for expert in sorted(
                    state.logical_to_slot,
                    key=lambda item: state.slots[
                        state.logical_to_slot[item]
                    ].last_used_at,
                    reverse=True,
                )
                if expert not in ranked
            )
            retained = ranked[:capacity]
            retained_slots = {state.logical_to_slot[expert] for expert in retained}
            for expert in list(state.logical_to_slot):
                if expert not in retained:
                    self._remove_mapping(
                        state, state.slots[state.logical_to_slot[expert]]
                    )
            available = [
                slot.slot for slot in state.slots if slot.slot not in retained_slots
            ]
            state.active_slots = retained_slots | set(
                available[: capacity - len(retained_slots)]
            )
            state.generation += 1
            state.counters.capacity_change_count += 1
            state.counters.capacity_change_time_ns += (
                time.perf_counter_ns() - started
            )
            state.sink.publish_mapping(dict(state.logical_to_slot), state.generation)
            return self.layer_status(layer)

    def resize(
        self,
        layer: int,
        slot_count: int,
        retain: list[int] | None = None,
        preserve_retained: bool = True,
    ) -> dict[str, Any]:
        state = self._layer(layer)
        if slot_count <= 0 or slot_count > state.global_experts:
            raise ValueError("slot count must be within the logical expert range")
        with state.lock:
            retained = list(
                dict.fromkeys(
                    retain
                    if retain is not None
                    else sorted(
                        state.logical_to_slot,
                        key=lambda item: state.slots[
                            state.logical_to_slot[item]
                        ].last_used_at,
                        reverse=True,
                    )
                )
            )[:slot_count]
            for expert in retained:
                if not self.source.contains(layer, expert):
                    raise FileNotFoundError(
                        f"cannot resize layer {layer}: backing object {expert} missing"
                    )

            retained_slots = [
                (expert, state.logical_to_slot[expert])
                for expert in retained
                if expert in state.logical_to_slot
            ]
            started = time.perf_counter_ns()
            state.sink.publish_mapping({}, state.generation + 1)
            fast_resize = getattr(state.sink, "resize_with_retained", None)
            state.logical_to_slot.clear()
            state.generation += 1
            state.counters.resize_count += 1
            try:
                if callable(fast_resize) and preserve_retained:
                    outcome = fast_resize(slot_count, retained_slots)
                    mapping = {
                        int(logical): int(slot)
                        for logical, slot in dict(outcome.get("mapping", {})).items()
                    }
                    state.slots = [
                        SlotRecord(slot=index) for index in range(slot_count)
                    ]
                    state.active_slots = set(range(slot_count))
                    now = time.monotonic()
                    for logical, slot_index in mapping.items():
                        slot = state.slots[slot_index]
                        slot.state = SlotState.READY
                        slot.logical_expert = logical
                        slot.last_used_at = now
                        slot.loads = 1
                    state.logical_to_slot = mapping
                    state.counters.gpu_copy_bytes += int(
                        outcome.get("bytes_copied", 0)
                    )
                    state.sink.finish_resize()
                    state.sink.publish_mapping(mapping, state.generation)
                else:
                    state.sink.begin_resize(slot_count)
                    state.slots = [
                        SlotRecord(slot=index) for index in range(slot_count)
                    ]
                    state.active_slots = set(range(slot_count))
                    if retained:
                        self.ensure(layer, retained, reason="resize_restore")
                    state.sink.finish_resize()
            except Exception:
                state.sink.publish_mapping({}, state.generation)
                raise
            finally:
                state.counters.resize_time_ns += time.perf_counter_ns() - started
            return self.layer_status(layer)

    def is_fully_resident(self, layer: int) -> bool:
        state = self._layer(layer)
        with state.lock:
            return bool(
                len(state.slots) == state.global_experts
                and len(state.logical_to_slot) == state.global_experts
                and all(slot.state == SlotState.READY for slot in state.slots)
            )

    def layers(self) -> list[int]:
        with self._lock:
            return sorted(self._layers)

    def status(self, include_mappings: bool = False) -> dict[str, Any]:
        return {
            "backend": "exact_expert_slot_dataplane",
            "data_plane_ready": bool(self._layers),
            "exact_route_required": True,
            "layers": [
                self.layer_status(layer, include_mapping=include_mappings)
                for layer in self.layers()
            ],
        }

    def layer_status(
        self, layer: int, include_mapping: bool = True
    ) -> dict[str, Any]:
        state = self._layer(layer)
        with state.lock:
            ready = sum(
                slot.state == SlotState.READY and slot.slot in state.active_slots
                for slot in state.slots
            )
            physical_ready = sum(slot.state == SlotState.READY for slot in state.slots)
            failed = sum(slot.state == SlotState.FAILED for slot in state.slots)
            average_us = (
                state.counters.load_time_ns / state.counters.misses / 1000
                if state.counters.misses
                else 0.0
            )
            payload = {
                "layer": layer,
                "global_experts": state.global_experts,
                "slot_count": len(state.slots),
                "active_slot_count": len(state.active_slots),
                "ready_slots": ready,
                "physical_ready_slots": physical_ready,
                "failed_slots": failed,
                "generation": state.generation,
                "counters": {
                    **asdict(state.counters),
                    "average_load_us": round(average_us, 3),
                },
            }
            if include_mapping:
                payload["logical_to_slot"] = dict(state.logical_to_slot)
            return payload

    def _layer(self, layer: int) -> LayerSlotState:
        with self._lock:
            state = self._layers.get(layer)
        if state is None:
            raise KeyError(f"layer {layer} is not registered")
        return state

    @staticmethod
    def _select_slot(
        state: LayerSlotState,
        protected: set[int],
        preferred: set[int] | None = None,
    ) -> SlotRecord:
        for slot in state.slots:
            if (
                slot.slot in state.active_slots
                and slot.state in {SlotState.EMPTY, SlotState.FAILED}
            ):
                return slot
        protected_preferred = preferred if preferred is not None else protected
        candidates = [
            slot
            for slot in state.slots
            if slot.slot in state.active_slots
            and slot.state == SlotState.READY
            and slot.logical_expert not in protected_preferred
        ]
        if not candidates and protected_preferred != protected:
            candidates = [
                slot
                for slot in state.slots
                if slot.slot in state.active_slots
                and slot.state == SlotState.READY
                and slot.logical_expert not in protected
            ]
        if not candidates:
            raise RuntimeError(
                f"layer {state.layer} has no evictable slot for exact routing"
            )
        return min(candidates, key=lambda item: item.last_used_at)

    @staticmethod
    def _remove_mapping(state: LayerSlotState, slot: SlotRecord) -> None:
        if slot.logical_expert is None:
            return
        state.logical_to_slot.pop(slot.logical_expert, None)
        state.counters.evictions += 1
        state.sink.invalidate(slot.slot)
        slot.logical_expert = None
        slot.state = SlotState.EMPTY
