from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.extractor import GraphExtractor
from graphrag.graph.relationship import Relationship

class GraphBuilder:
    def __init__(self,extractor:GraphExtractor):
        self.extractor=extractor
    
    def add_text(self, graph: KnowledgeGraph, text: str) -> None:
        entities, relationships = self.extractor.extract(text)

        entity_id_map = {}
        for entity in entities:
            extracted_id = entity.id
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
                    metadata=relationship.metadata.copy(),
                )
            )

        errors = graph.validate()
        if errors:
            raise ValueError("Invalid knowledge graph: " + "; ".join(errors))
