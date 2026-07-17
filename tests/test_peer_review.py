from __future__ import annotations

from pathlib import Path

import pytest

from pllm.peer_review import (
    AuthorAgent,
    PeerReviewLoop,
    ReviewerAgent,
    parse_author_output,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if "Reviewer Agent" in system:
            return f"总体判断：第 {self.calls} 次审查\n\n建议结论：Borderline"
        round_number = (self.calls + 1) // 2
        return (
            "<REBUTTAL>接受并修改。</REBUTTAL>"
            f"<REVISION># Revision {round_number}\n\nNo fabricated results.</REVISION>"
        )


def test_parse_author_output_rejects_unstructured_text() -> None:
    with pytest.raises(ValueError):
        parse_author_output("plain response")


def test_peer_review_loop_requires_three_rounds(tmp_path: Path) -> None:
    client = FakeClient()
    loop = PeerReviewLoop(
        ReviewerAgent(client, "PLLM Reviewer Agent"),
        AuthorAgent(client, "PLLM Rebuttal Agent"),
    )
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("# Draft\n", encoding="utf-8")
    with pytest.raises(ValueError):
        loop.run(manuscript, [], tmp_path / "reviews", rounds=2)


def test_peer_review_loop_writes_all_rounds_without_overwrite(tmp_path: Path) -> None:
    client = FakeClient()
    loop = PeerReviewLoop(
        ReviewerAgent(client, "PLLM Reviewer Agent"),
        AuthorAgent(client, "PLLM Rebuttal Agent"),
    )
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("# Draft\n", encoding="utf-8")
    output_dir = tmp_path / "reviews"

    final_path = loop.run(manuscript, [], output_dir, rounds=4)

    assert client.calls == 8
    assert manuscript.read_text(encoding="utf-8") == "# Draft\n"
    assert "Revision 4" in final_path.read_text(encoding="utf-8")
    for round_number in range(1, 5):
        assert (output_dir / f"round-{round_number}-review.md").exists()
        assert (output_dir / f"round-{round_number}-rebuttal.md").exists()
        assert (output_dir / f"manuscript-round-{round_number}.md").exists()
