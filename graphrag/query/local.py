"""Local GraphRAG queries: retrieve a relevant entity neighbourhood before answering."""

from dataclasses import dataclass
import re

from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.relationship import Relationship
from graphrag.llm.client import LLM


@dataclass
class LocalQueryResult:
    answer: str
    entity_ids: list[str]
    relationships: list[Relationship]


class LocalQueryEngine:
    def __init__(self, llm: LLM, max_entities: int = 3):
        self.llm = llm
        self.max_entities = max_entities

    def query(self, graph: KnowledgeGraph, question: str) -> LocalQueryResult:
        seed_ids = self._find_relevant_entities(graph, question)
        relationship_map = {
            relationship.id: relationship
            for relationship in graph.relationships_for_entities(set(seed_ids))
        }
        entity_ids = set(seed_ids)
        for relationship in relationship_map.values():
            entity_ids.add(relationship.source)
            entity_ids.add(relationship.target)
        ordered_entity_ids = sorted(entity_ids)
        relationships = list(relationship_map.values())
        context = self._format_context(graph, ordered_entity_ids, relationships)
        prompt = f"""LOCAL QUERY
Answer the question using the local knowledge-graph context. If the context does not
contain the answer, say so instead of inventing facts.

Question: {question}

Context:
{context}
"""
        return LocalQueryResult(
            answer=self.llm.generate(prompt).strip(),
            entity_ids=ordered_entity_ids,
            relationships=relationships,
        )

    def _find_relevant_entities(self, graph: KnowledgeGraph, question: str) -> list[str]:
        query_terms = set(re.findall(r"\w+", question.lower()))
        scores = []
        for entity in graph.entities.values():
            searchable_text = f"{entity.name} {entity.description}".lower()
            score = sum(term in searchable_text for term in query_terms)
            if entity.name.lower() in question.lower():
                score += 10
            if score:
                scores.append((score, entity.id))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [entity_id for _, entity_id in scores[: self.max_entities]]

    @staticmethod
    def _format_context(
        graph: KnowledgeGraph,
        entity_ids: list[str],
        relationships: list[Relationship],
    ) -> str:
        entities = "\n".join(
            f"- {graph.entities[entity_id].name}: {graph.entities[entity_id].description}"
            for entity_id in entity_ids
        ) or "- No relevant entities found"
        edges = "\n".join(
            f"- {graph.entities[relationship.source].name} -> "
            f"{graph.entities[relationship.target].name}: {relationship.description}"
            for relationship in relationships
        ) or "- No connected relationships found"
        return f"Entities:\n{entities}\nRelationships:\n{edges}"
