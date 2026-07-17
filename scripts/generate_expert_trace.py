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
from pllm.expert_trace import synthetic_trace, write_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a no-GPU expert route trace")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "expert_trace_synthetic.jsonl",
    )
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    catalog = ExpertCatalog.from_model(args.model)
    count = write_trace(
        args.output,
        synthetic_trace(catalog, args.requests, args.tokens, args.seed),
        catalog,
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "records": count,
                "source": "synthetic_no_gpu",
                "real_route_evidence": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
