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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MoE expert objects without loading model tensors"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "expert_catalog.json"
    )
    parser.add_argument("--include-experts", action="store_true")
    args = parser.parse_args()

    catalog = ExpertCatalog.from_model(args.model)
    payload = catalog.summary(include_experts=args.include_experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
