import json
from graphrag.graph.entity import Entity
from graphrag.graph.relationship import Relationship

def parse_extraction_response(response: str) -> tuple[list[Entity], list[Relationship]]:
    data = json.loads(response)
    if not isinstance(data, dict):
        raise ValueError("Extraction response must be a JSON object")
    if "entities" not in data or "relationships" not in data:
        raise ValueError("Extraction response requires entities and relationships")

    entities = []
    relationships = []

    for index, item in enumerate(data["entities"]):
        entities.append(
            Entity(
                id=f"entity_{index + 1}",
                name=item["name"],
                entity_type=item["type"],
                description=item.get("description", ""),
            )
        )

    entity_ids = {
        entity.name.lower(): entity.id
        for entity in entities
    }

    for index, item in enumerate(data["relationships"]):
        source = item["source"].lower()
        target = item["target"].lower()
        if source not in entity_ids or target not in entity_ids:
            raise ValueError("Relationship references an entity that was not extracted")
        relationships.append(
            Relationship(
                id=f"rel_{index + 1}",
                source=entity_ids[source],
                target=entity_ids[target],
                description=item.get("description", ""),
                weight=float(item.get("weight", 1.0)),
            )
        )

    return entities, relationships
