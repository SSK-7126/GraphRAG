
import json
import re

from graphrag.graph.entity import Entity
from graphrag.graph.relationship import Relationship


def normalize_name(name: str) -> str:
    """Normalize names to handle case and whitespace differences."""
    return " ".join(name.strip().casefold().split())


def parse_extraction_response(
    response: str,
) -> tuple[list[Entity], list[Relationship]]:

    data = _parse_json_object(response)

    if not isinstance(data, dict):
        raise ValueError("Extraction response must be a JSON object")

    if "entities" not in data or "relationships" not in data:
        raise ValueError(
            "Extraction response requires entities and relationships"
        )

    entities = []
    relationships = []

    entities = parse_entities(data.get("entities", []))

    # Map normalized entity names to IDs
    entity_ids = {
        normalize_name(entity.name): entity.id
        for entity in entities
    }

    # Parse relationships
    for index, item in enumerate(data["relationships"]):
        source = normalize_name(item["source"])
        target = normalize_name(item["target"])

        if source not in entity_ids or target not in entity_ids:
            missing = []

            if source not in entity_ids:
                missing.append(item["source"])

            if target not in entity_ids:
                missing.append(item["target"])

            print(
                f"WARNING: Skipping relationship {index + 1}. "
                f"Missing entities: {missing}"
            )
            continue

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


def parse_entities(items: object) -> list[Entity]:
    """Validate the entity part of a model response without dropping bad rows."""
    if not isinstance(items, list):
        raise ValueError("Extraction response field 'entities' must be an array")
    entities: list[Entity] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"Entity {index + 1} has no valid name")
        name = item["name"].strip()
        if not name:
            raise ValueError(f"Entity {index + 1} has an empty name")
        entity_type = item.get("type", "unknown")
        if not isinstance(entity_type, str):
            entity_type = "unknown"
        entities.append(Entity(
            id=f"entity_{index + 1}", name=name,
            entity_type=entity_type.strip() or "unknown",
            description=str(item.get("description", "")).strip(),
        ))
    return entities


def parse_relationship_response(response: str, entities: list[Entity]) -> list[Relationship]:
    """Parse a relationship-only response against entities from the same chunk.

    Invalid edges are returned through ``rejections`` when supplied.  They are
    never inserted with a dangling endpoint, but one bad edge does not discard
    the other supported facts in a chunk.
    """
    return parse_relationship_response_with_rejections(response, entities)[0]


def parse_relationship_response_with_rejections(
    response: str, entities: list[Entity]
) -> tuple[list[Relationship], list[str]]:
    data = _parse_json_object(response)
    if "relationships" not in data or not isinstance(data["relationships"], list):
        raise ValueError("Relationship response requires a 'relationships' array")
    entity_ids = {normalize_name(entity.name): entity.id for entity in entities}
    aliases = {_entity_alias(entity.name): entity.id for entity in entities}
    relationships: list[Relationship] = []
    rejections: list[str] = []
    for index, item in enumerate(data["relationships"]):
        if not isinstance(item, dict):
            raise ValueError(f"Relationship {index + 1} must be an object")
        source_name = item.get("source")
        target_name = item.get("target")
        if not isinstance(source_name, str) or not isinstance(target_name, str):
            raise ValueError(f"Relationship {index + 1} requires string source and target")
        source = entity_ids.get(normalize_name(source_name)) or aliases.get(_entity_alias(source_name))
        target = entity_ids.get(normalize_name(target_name)) or aliases.get(_entity_alias(target_name))
        if source is None or target is None:
            message = (
                f"Relationship {index + 1} refers to an entity absent from this chunk: "
                f"{source_name!r} -> {target_name!r}"
            )
            rejections.append(message)
            continue
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Relationship {index + 1} has an invalid weight") from error
        relationships.append(Relationship(
            id=f"rel_{index + 1}", source=source, target=target,
            description=str(item.get("description", "")).strip(), weight=weight,
        ))
    return relationships, rejections


def _entity_alias(name: str) -> str:
    """Normalize harmless title and punctuation variants, without fuzzy matching."""
    alias = normalize_name(name)
    alias = re.sub(r"^(?:mr|mrs|ms|dr|prof|rep|representative|sen|senator)\.?\s+", "", alias)
    return re.sub(r"[^\w]+", "", alias)


def _parse_json_object(response: str) -> dict:
    """Accept JSON returned in a markdown fence while still rejecting non-JSON prose."""
    cleaned = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON extraction response: {error.msg}") from error
    if not isinstance(data, dict):
        raise ValueError("Extraction response must be a JSON object")
    return data
