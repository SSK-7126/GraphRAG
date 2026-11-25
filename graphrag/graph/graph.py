from graphrag.graph.relationship import Relationship
from graphrag.graph.entity import Entity
from graphrag.graph.parser import normalize_name


class KnowledgeGraph:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}

    def add_entity(self, entity: Entity) -> Entity:
        for existing_entity in self.entities.values():
            if normalize_name(existing_entity.name) == normalize_name(entity.name):
                self._merge_metadata(existing_entity.metadata, entity.metadata)
                if not existing_entity.description and entity.description:
                    existing_entity.description = entity.description
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
            existing_chunks = set(existing_relationship.metadata.get("source_chunks", []))
            incoming_chunks = set(relationship.metadata.get("source_chunks", []))
            if (
                existing_relationship.source == relationship.source
                and existing_relationship.target == relationship.target
                and (
                    normalize_name(existing_relationship.description)
                    == normalize_name(relationship.description)
                    or bool(existing_chunks & incoming_chunks)
                )
            ):
                self._merge_metadata(existing_relationship.metadata, relationship.metadata)
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

    @staticmethod
    def _merge_metadata(existing: dict, incoming: dict) -> None:
        """Preserve provenance while avoiding repeated values on resumed runs."""
        for key, value in incoming.items():
            if key in {"source_documents", "source_chunks"}:
                current = existing.setdefault(key, [])
                for item in value if isinstance(value, list) else [value]:
                    if item not in current:
                        current.append(item)
            elif key not in existing:
                existing[key] = value
