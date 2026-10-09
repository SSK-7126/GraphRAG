"""Create LLM-readable summaries of graph communities."""

from dataclasses import dataclass

from graphrag.graph.community import Community
from graphrag.graph.graph import KnowledgeGraph
from graphrag.llm.client import LLM


@dataclass
class CommunitySummary:
    community_id: str
    summary: str
    entity_ids: list[str]


class CommunitySummarizer:
    def __init__(self, llm: LLM):
        self.llm = llm

    def summarize(
        self, graph: KnowledgeGraph, communities: list[Community]
    ) -> list[CommunitySummary]:
        return [self.summarize_community(graph, community) for community in communities]

    def summarize_community(
        self, graph: KnowledgeGraph, community: Community
    ) -> CommunitySummary:
        entity_ids = set(community.entity_ids)
        entities = [graph.entities[entity_id] for entity_id in community.entity_ids]
        relationships = [
            relationship
            for relationship in graph.relationships.values()
            if relationship.source in entity_ids and relationship.target in entity_ids
        ]
        entity_context = "\n".join(
            f"- {entity.name} ({entity.entity_type}): {entity.description}"
            for entity in entities
        ) or "- No entities"
        relationship_context = "\n".join(
            f"- {graph.entities[relationship.source].name} -> "
            f"{graph.entities[relationship.target].name}: {relationship.description}"
            for relationship in relationships
        ) or "- No relationships"
        prompt = f"""COMMUNITY SUMMARY
Summarize this knowledge-graph community. Describe its main theme and the important
relationships. Use only the supplied context and be concise.

Entities:
{entity_context}

Relationships:
{relationship_context}
"""
        return CommunitySummary(
            community_id=community.id,
            summary=self.llm.generate(prompt).strip(),
            entity_ids=community.entity_ids.copy(),
        )
