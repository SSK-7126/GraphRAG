from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.extractor import GraphExtractor
from graphrag.graph.relationship import Relationship

class GraphBuilder:
    def __init__(self,extractor:GraphExtractor):
        self.extractor=extractor
    
    def add_text(self, graph: KnowledgeGraph, text: str) -> None:
        entities, relationships = self.extractor.extract(text)
        self.add_extraction(graph, entities, relationships)

    def add_extraction(
        self,
        graph: KnowledgeGraph,
        entities,
        relationships,
        *,
        document_id: str | None = None,
        chunk_id: str | None = None,
    ) -> tuple[int, int]:
        """Insert one chunk's extraction and retain document/chunk provenance."""
        provenance = {
            "source_documents": [document_id] if document_id else [],
            "source_chunks": [chunk_id] if chunk_id else [],
        }

        entity_id_map = {}
        for entity in entities:
            extracted_id = entity.id
            entity.metadata = {**entity.metadata, **provenance}
            existing_entity = graph.add_entity(entity)
            entity_id_map[extracted_id] = existing_entity.id

        for relationship in relationships:
            if relationship.source not in entity_id_map:
                raise ValueError(f"Unknown relationship source: {relationship.source}")
            if relationship.target not in entity_id_map:
                raise ValueError(f"Unknown relationship target: {relationship.target}")

            graph.add_relationship(
                Relationship(
                    id=relationship.id,
                    source=entity_id_map[relationship.source],
                    target=entity_id_map[relationship.target],
                    description=relationship.description,
                    weight=relationship.weight,
                    metadata={**relationship.metadata, **provenance},
                )
            )

        errors = graph.validate()
        if errors:
            raise ValueError("Invalid knowledge graph: " + "; ".join(errors))
        return len(entities), len(relationships)

    @staticmethod
    def normalize_entities(entities) -> None:
        """Apply deterministic whitespace normalization before graph-level deduplication."""
        for entity in entities:
            entity.name = " ".join(entity.name.split())
            entity.entity_type = " ".join(entity.entity_type.split()) or "unknown"
            entity.description = " ".join(entity.description.split())
