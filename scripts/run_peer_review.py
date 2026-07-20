from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pllm.peer_review import (
    AuthorAgent,
    OpenAICompatibleClient,
    PeerReviewLoop,
    ReviewerAgent,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run iterative PLLM reviewer/rebuttal agents"
    )
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=ROOT / "docs" / "PLLM项目报告.md",
    )
    parser.add_argument("--context", action="append", type=Path, default=[])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "peer-review"
    )
    parser.add_argument("--apply-final", action="store_true")
    args = parser.parse_args()

    context_paths = args.context or [
        ROOT / "docs" / "主流推理框架暂停恢复调研.md",
    ]
    client = OpenAICompatibleClient.from_environment()
    reviewer = ReviewerAgent(
        client=client,
        system_prompt=(ROOT / "agents" / "reviewer.system.md").read_text(
            encoding="utf-8"
        ),
    )
    author = AuthorAgent(
        client=client,
        system_prompt=(ROOT / "agents" / "author.system.md").read_text(
            encoding="utf-8"
        ),
    )
    final_path = PeerReviewLoop(reviewer, author).run(
        manuscript_path=args.manuscript,
        context_paths=context_paths,
        output_dir=args.output_dir,
        rounds=args.rounds,
        apply_final=args.apply_final,
    )
    print(final_path)


if __name__ == "__main__":
    main()
