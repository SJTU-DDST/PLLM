from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pllm.config import DEFAULT_MODEL_PATH
from pllm.expert_catalog import ExpertCatalog
from pllm.expert_residency import (
    ConformalExpertPredictor,
    ExpertCacheSimulator,
    ResidencyPlanner,
    ResourceEnvelope,
    RouteHistoryPredictor,
)
from pllm.expert_trace import read_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a no-GPU exact expert residency simulation"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "results" / "expert_trace_synthetic.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "expert_residency_simulation.json",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    catalog = ExpertCatalog.from_model(args.model)
    records = list(read_trace(args.trace, catalog))
    request_ids = list(dict.fromkeys(record.request_id for record in records))
    if len(request_ids) < 3:
        raise ValueError("trace requires at least three request IDs for train/cal/test")
    train_ids = set(request_ids[:-2])
    calibration_id = request_ids[-2]
    test_id = request_ids[-1]
    train = [record for record in records if record.request_id in train_ids]
    calibration = [
        record for record in records if record.request_id == calibration_id
    ]
    test = [record for record in records if record.request_id == test_id]

    base = RouteHistoryPredictor(catalog.experts_per_layer)
    base.fit(train)
    predictor = ConformalExpertPredictor(base, alpha=args.alpha)
    calibration_report = predictor.calibrate(calibration)
    test_report = predictor.evaluate(test)

    simulations = []
    for slots in (32, 64, 128, 256):
        simulation = ExpertCacheSimulator(catalog, predictor, slots).run(test)
        simulations.append(simulation.to_dict())

    reference = next(item for item in simulations if item["slots_per_layer"] == 128)
    records_per_token = len(catalog.moe_layers)
    false_bytes_per_token = (
        reference["false_prefetch_bytes_per_record"] * records_per_token
    )
    planner = ResidencyPlanner(catalog)
    scenarios = {
        "idle": ResourceEnvelope(foreground_reserve_gib=20, compute_duty_cycle=1.0),
        "creative": ResourceEnvelope(
            foreground_reserve_gib=64,
            io_budget_gib_s=2.0,
            compute_duty_cycle=0.35,
            requested_token_rate=5.0,
        ),
        "memory_emergency": ResourceEnvelope(
            foreground_reserve_gib=104,
            io_budget_gib_s=0.5,
            compute_duty_cycle=0.1,
        ),
    }
    plans = {
        "idle_observed": planner.plan(
            scenarios["idle"], byte_hit_rate=1.0, false_prefetch_bytes_per_token=0
        ).to_dict(),
        "creative_hypothetical_in_domain": planner.plan(
            scenarios["creative"],
            byte_hit_rate=0.95,
            false_prefetch_bytes_per_token=(
                catalog.active_expert_bytes_per_token * 0.05
            ),
        ).to_dict(),
        "creative_observed_domain_shift": planner.plan(
            scenarios["creative"],
            byte_hit_rate=float(reference["byte_hit_rate"]),
            false_prefetch_bytes_per_token=false_bytes_per_token,
        ).to_dict(),
        "memory_emergency": planner.plan(
            scenarios["memory_emergency"],
            byte_hit_rate=float(reference["byte_hit_rate"]),
            false_prefetch_bytes_per_token=false_bytes_per_token,
        ).to_dict(),
    }
    plans["creative_hypothetical_in_domain"]["evidence"] = (
        "hypothetical_control_input_not_model_measurement"
    )
    payload = {
        "schema_version": 1,
        "trace_source": "synthetic_no_gpu",
        "real_route_evidence": False,
        "dataset_split": {
            "train_requests": sorted(train_ids),
            "calibration_request": calibration_id,
            "test_request": test_id,
        },
        "calibration": calibration_report.to_dict(),
        "test": test_report,
        "cache_simulations": simulations,
        "plans": plans,
        "claims_allowed": [
            "schema and accounting invariants",
            "exact-route fallback behavior",
            "control-plane phase transitions",
        ],
        "claims_forbidden": [
            "Nemotron predictor accuracy",
            "runtime memory reclaim",
            "real SSD or RDMA throughput",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
