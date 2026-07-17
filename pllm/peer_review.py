from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class OpenAICompatibleClient:
    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 600.0
    max_tokens: int = 16_384

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient":
        base_url = os.environ.get("PLLM_REVIEW_BASE_URL", "").rstrip("/")
        model = os.environ.get("PLLM_REVIEW_MODEL", "")
        if not base_url or not model:
            raise ValueError(
                "PLLM_REVIEW_BASE_URL and PLLM_REVIEW_MODEL are required"
            )
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("PLLM_REVIEW_API_KEY", ""),
        )

    def complete(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])


@dataclass
class ReviewerAgent:
    client: ChatClient
    system_prompt: str

    def review(self, manuscript: str, context: str, round_number: int) -> str:
        return self.client.complete(
            self.system_prompt,
            f"""这是第 {round_number} 轮审稿。

以下材料只用于核验，不代表其中主张已经成立：

<CONTEXT>
{context}
</CONTEXT>

请审查当前论文：

<MANUSCRIPT>
{manuscript}
</MANUSCRIPT>
""",
        ).strip()


@dataclass
class AuthorAgent:
    client: ChatClient
    system_prompt: str

    def rebut_and_revise(
        self, manuscript: str, review: str, context: str, round_number: int
    ) -> tuple[str, str]:
        output = self.client.complete(
            self.system_prompt,
            f"""这是第 {round_number} 轮答辩与修订。

<CONTEXT>
{context}
</CONTEXT>

<CURRENT_MANUSCRIPT>
{manuscript}
</CURRENT_MANUSCRIPT>

<REVIEW>
{review}
</REVIEW>
""",
        )
        return parse_author_output(output)


def parse_author_output(output: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"\s*<REBUTTAL>\s*(.*?)\s*</REBUTTAL>\s*"
        r"<REVISION>\s*(.*?)\s*</REVISION>\s*",
        output,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("author output does not follow the rebuttal/revision protocol")
    rebuttal, revision = (part.strip() for part in match.groups())
    if not rebuttal or not revision:
        raise ValueError("author output contains an empty rebuttal or revision")
    return rebuttal, revision


@dataclass
class PeerReviewLoop:
    reviewer: ReviewerAgent
    author: AuthorAgent

    def run(
        self,
        manuscript_path: Path,
        context_paths: list[Path],
        output_dir: Path,
        rounds: int = 4,
        apply_final: bool = False,
    ) -> Path:
        if rounds < 3:
            raise ValueError("at least three review rounds are required")
        manuscript = manuscript_path.read_text(encoding="utf-8")
        context = "\n\n".join(
            f"# Context: {path}\n\n{path.read_text(encoding='utf-8')}"
            for path in context_paths
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write(output_dir / "manuscript-round-0.md", manuscript)

        for round_number in range(1, rounds + 1):
            review = self.reviewer.review(manuscript, context, round_number)
            rebuttal, manuscript = self.author.rebut_and_revise(
                manuscript, review, context, round_number
            )
            self._write(output_dir / f"round-{round_number}-review.md", review)
            self._write(output_dir / f"round-{round_number}-rebuttal.md", rebuttal)
            self._write(
                output_dir / f"manuscript-round-{round_number}.md", manuscript
            )

        final_path = output_dir / "manuscript-final.md"
        self._write(final_path, manuscript)
        if apply_final:
            self._write(manuscript_path, manuscript)
        return final_path

    @staticmethod
    def _write(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
        temporary.replace(path)
