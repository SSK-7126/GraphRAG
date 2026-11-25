from graphrag.graph.relationship import Relationship
from graphrag.graph.entity import Entity
from graphrag.graph.parser import normalize_name


class KnowledgeGraph:
    def __init__(self, *, enable_indexes: bool = True):
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}
        # The baseline runner can deliberately disable these indexes to retain
        # the original scan-based implementation for apples-to-apples studies.
        self.enable_indexes = enable_indexes
        self._entity_name_index: dict[str, str] = {}
        self._relationships_by_endpoint: dict[tuple[str, str], list[str]] = {}
        self._relationship_description_index: dict[tuple[str, str, str], str] = {}
        self._relationship_chunk_index: dict[tuple[str, str, str], str] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._revision = 0

    def add_entity(self, entity: Entity) -> Entity:
        if self.enable_indexes:
            name = normalize_name(entity.name)
            existing_id = self._entity_name_index.get(name)
            if existing_id is not None:
                existing_entity = self.entities[existing_id]
                self._merge_metadata(existing_entity.metadata, entity.metadata)
                if not existing_entity.description and entity.description:
                    existing_entity.description = entity.description
                    self._revision += 1
                return existing_entity
        for existing_entity in self.entities.values():
            if normalize_name(existing_entity.name) == normalize_name(entity.name):
                self._merge_metadata(existing_entity.metadata, entity.metadata)
                if not existing_entity.description and entity.description:
                    existing_entity.description = entity.description
                return existing_entity

        entity.id = self._available_id(entity.id, self.entities)
        self.entities[entity.id] = entity
        if self.enable_indexes:
            self._entity_name_index[normalize_name(entity.name)] = entity.id
            self._adjacency.setdefault(entity.id, set())
        self._revision += 1
        return entity

    def add_relationship(self, relationship: Relationship) -> Relationship:
        if relationship.source not in self.entities:
            raise ValueError(f"Relationship source does not exist: {relationship.source}")
        if relationship.target not in self.entities:
            raise ValueError(f"Relationship target does not exist: {relationship.target}")

        if self.enable_indexes:
            endpoint = (relationship.source, relationship.target)
            matching_id = self._relationship_description_index.get(
                (*endpoint, normalize_name(relationship.description))
            )
            if matching_id is None:
                matching_id = next((
                    self._relationship_chunk_index.get((*endpoint, chunk_id))
                    for chunk_id in relationship.metadata.get("source_chunks", [])
                    if self._relationship_chunk_index.get((*endpoint, chunk_id)) is not None
                ), None)
            if matching_id is not None:
                existing_relationship = self.relationships[matching_id]
                self._merge_metadata(existing_relationship.metadata, relationship.metadata)
                # New provenance may itself become a future duplicate key.
                self._index_relationship(existing_relationship, append=False)
                return existing_relationship
        candidates = (
            (self.relationships[relationship_id] for relationship_id in self._relationships_by_endpoint.get((relationship.source, relationship.target), []))
            if self.enable_indexes
            else self.relationships.values()
        )
        for existing_relationship in candidates:
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
        if self.enable_indexes:
            self._index_relationship(relationship)
        self._revision += 1
        return relationship

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        return self.relationships.get(relationship_id)

    def relationships_for_entities(self, entity_ids: set[str]) -> list[Relationship]:
        if self.enable_indexes:
            relationship_ids: set[str] = set()
            for entity_id in entity_ids:
                relationship_ids.update(self._adjacency.get(entity_id, ()))
            return [self.relationships[relationship_id] for relationship_id in sorted(relationship_ids)]
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

    def relationships_within_entities(self, entity_ids: set[str]) -> list[Relationship]:
        """Return internal edges without scanning every edge for each community."""
        candidates = self.relationships_for_entities(entity_ids)
        return [
            relationship for relationship in candidates
            if relationship.source in entity_ids and relationship.target in entity_ids
        ]

    def rebuild_indexes(self) -> None:
        """Reconstruct derived indexes after direct persistence loading."""
        self._entity_name_index.clear()
        self._relationships_by_endpoint.clear()
        self._relationship_description_index.clear()
        self._relationship_chunk_index.clear()
        self._adjacency = {entity_id: set() for entity_id in self.entities}
        if not self.enable_indexes:
            return
        for entity in self.entities.values():
            self._entity_name_index.setdefault(normalize_name(entity.name), entity.id)
        for relationship in self.relationships.values():
            self._index_relationship(relationship)

    def _index_relationship(self, relationship: Relationship, *, append: bool = True) -> None:
        endpoint = (relationship.source, relationship.target)
        if append:
            self._relationships_by_endpoint.setdefault(endpoint, []).append(relationship.id)
        self._relationship_description_index.setdefault(
            (*endpoint, normalize_name(relationship.description)), relationship.id
        )
        for chunk_id in relationship.metadata.get("source_chunks", []):
            self._relationship_chunk_index.setdefault((*endpoint, chunk_id), relationship.id)
        self._adjacency.setdefault(relationship.source, set()).add(relationship.id)
        self._adjacency.setdefault(relationship.target, set()).add(relationship.id)

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
