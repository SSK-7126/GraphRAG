"""Global GraphRAG queries using a small map/reduce flow over community summaries."""

from dataclasses import dataclass

from graphrag.graph.community_summarizer import CommunitySummary
from graphrag.llm.client import LLM


@dataclass
class GlobalQueryResult:
    answer: str
    partial_answers: list[str]


class GlobalQueryEngine:
    def __init__(self, llm: LLM):
        self.llm = llm

    def query(
        self, community_summaries: list[CommunitySummary], question: str
    ) -> GlobalQueryResult:
        partial_answers = [
            self.llm.generate(
                f"""GLOBAL MAP
Question: {question}
Community summary: {community.summary}
Give a short answer using only this community summary."""
            ).strip()
            for community in community_summaries
        ]
        evidence = "\n".join(
            f"- Community {summary.community_id}: {answer}"
            for summary, answer in zip(community_summaries, partial_answers)
        ) or "- No community summaries are available"
        answer = self.llm.generate(
            f"""GLOBAL REDUCE
Question: {question}
Combine the following community-level answers into one coherent high-level answer.
Do not add facts that are absent from the evidence.

Evidence:
{evidence}"""
        ).strip()
        return GlobalQueryResult(answer=answer, partial_answers=partial_answers)
