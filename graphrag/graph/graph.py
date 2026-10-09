from graphrag.graph.relationship import Relationship
from graphrag.graph.entity import Entity


class KnowledgeGraph:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}

    def add_entity(self, entity: Entity) -> Entity:
        for existing_entity in self.entities.values():
            if existing_entity.name.lower() == entity.name.lower():
                return existing_entity

        entity.id = self._available_id(entity.id, self.entities)
        self.entities[entity.id] = entity
        return entity

    def add_relationship(self, relationship: Relationship) -> Relationship:
        if relationship.source not in self.entities:
            raise ValueError(f"Relationship source does not exist: {relationship.source}")
        if relationship.target not in self.entities:
            raise ValueError(f"Relationship target does not exist: {relationship.target}")

        for existing_relationship in self.relationships.values():
            if (
                existing_relationship.source == relationship.source
                and existing_relationship.target == relationship.target
                and existing_relationship.description == relationship.description
            ):
                return existing_relationship

        relationship.id = self._available_id(relationship.id, self.relationships)
        self.relationships[relationship.id] = relationship
        return relationship

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        return self.relationships.get(relationship_id)

    def relationships_for_entities(self, entity_ids: set[str]) -> list[Relationship]:
        return [
            relationship
            for relationship in self.relationships.values()
            if relationship.source in entity_ids or relationship.target in entity_ids
        ]

    def validate(self) -> list[str]:
        """Return validation errors; an empty list means the graph is valid."""
        errors = []
        for relationship in self.relationships.values():
            if relationship.source not in self.entities:
                errors.append(
                    f"Relationship {relationship.id} has missing source {relationship.source}"
                )
            if relationship.target not in self.entities:
                errors.append(
                    f"Relationship {relationship.id} has missing target {relationship.target}"
                )
        return errors

    @staticmethod
    def _available_id(preferred_id: str, items: dict[str, object]) -> str:
        if preferred_id not in items:
            return preferred_id

        suffix = 2
        while f"{preferred_id}_{suffix}" in items:
            suffix += 1
        return f"{preferred_id}_{suffix}"
