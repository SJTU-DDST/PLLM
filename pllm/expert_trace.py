from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .expert_catalog import ExpertCatalog


@dataclass(slots=True)
class ExpertRouteRecord:
    request_id: str
    workload: str
    phase: str
    token_index: int
    layer: int
    actual_experts: list[int]
    source: str = "runtime"

    def validate(self, catalog: ExpertCatalog) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(f"invalid phase: {self.phase}")
        if self.layer not in catalog.moe_layers:
            raise ValueError(f"layer {self.layer} is not a catalogued MoE layer")
        if len(self.actual_experts) != catalog.active_experts_per_token:
            raise ValueError(
                f"expected {catalog.active_experts_per_token} actual experts, "
                f"got {len(self.actual_experts)}"
            )
        if len(set(self.actual_experts)) != len(self.actual_experts):
            raise ValueError("actual_experts contains duplicates")
        if any(
            expert < 0 or expert >= catalog.experts_per_layer
            for expert in self.actual_experts
        ):
            raise ValueError("actual_experts contains an out-of-range expert id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpertRouteRecord":
        return cls(
            request_id=str(payload["request_id"]),
            workload=str(payload["workload"]),
            phase=str(payload["phase"]),
            token_index=int(payload["token_index"]),
            layer=int(payload["layer"]),
            actual_experts=[int(value) for value in payload["actual_experts"]],
            source=str(payload.get("source", "runtime")),
        )


def write_trace(
    path: str | Path,
    records: Iterable[ExpertRouteRecord],
    catalog: ExpertCatalog,
) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            record.validate(catalog)
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(output)
    return count


def read_trace(
    path: str | Path, catalog: ExpertCatalog
) -> Iterator[ExpertRouteRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record is not an object")
                record = ExpertRouteRecord.from_dict(payload)
                record.validate(catalog)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid trace record at line {line_number}") from exc
            yield record


def synthetic_trace(
    catalog: ExpertCatalog,
    requests: int = 3,
    tokens_per_request: int = 24,
    seed: int = 7,
) -> Iterator[ExpertRouteRecord]:
    if requests <= 0 or tokens_per_request <= 0:
        raise ValueError("requests and tokens_per_request must be positive")
    rng = random.Random(seed)
    top_k = catalog.active_experts_per_token
    num_experts = catalog.experts_per_layer
    workloads = ("code", "math", "chat", "rag")

    for request_index in range(requests):
        workload = workloads[request_index % len(workloads)]
        domain_start = (request_index * 73) % num_experts
        hot = [(domain_start + offset * 3) % num_experts for offset in range(64)]
        previous: dict[int, list[int]] = {}
        for token_index in range(tokens_per_request):
            for layer in catalog.moe_layers:
                retained = previous.get(layer, [])[: max(1, top_k // 2)]
                candidates = list(dict.fromkeys(retained + hot))
                rng.shuffle(candidates)
                selected = candidates[:top_k]
                while len(selected) < top_k:
                    candidate = rng.randrange(num_experts)
                    if candidate not in selected:
                        selected.append(candidate)
                previous[layer] = selected
                yield ExpertRouteRecord(
                    request_id=f"synthetic-{request_index}",
                    workload=workload,
                    phase="decode",
                    token_index=token_index,
                    layer=layer,
                    actual_experts=selected,
                    source="synthetic_no_gpu",
                )
