from graphrag.llm.client import LLM
from graphrag.graph.parser import (
    parse_entities,
    parse_extraction_response,
    parse_relationship_response_with_rejections,
)
from graphrag.graph.extractor import GraphExtractor
from graphrag.graph.entity import Entity
from graphrag.graph.relationship import Relationship

class LLMGraphExtractor(GraphExtractor):
    def __init__(self, llm: LLM):
        self.llm = llm
        self.last_rejections: list[str] = []

    def extract(self, text: str) -> tuple[list[Entity], list[Relationship]]:
        """Compatibility method used by the original small demo pipeline."""
        prompt = f"""Extract every explicitly stated entity and relationship
                    from the text below.

                    Text:
                    {text}

                    Return ONLY valid JSON in exactly this structure:
                        {{
                        "entities": [
                            {{
                            "name": "entity name",
                            "type": "person, company, product, or other",
                            "description": "short description"
                            }}
                        ],
                        "relationships": [
                            {{
                            "source": "source entity name",
                            "target": "target entity name",
                            "description": "relationship stated in the text",
                            "weight": 1.0
                            }}
                        ]
                        }}

                        Rules:
                        - Include every relationship explicitly stated in the text.
                        - Every relationship source and target must match an entity name.
                        - Do not invent facts or relationships.
                        - Use an empty array when there are no entities or relationships.
                        - Do not include Markdown fences or text outside the JSON.
                        """
        response = self.llm.generate(prompt)
        return parse_extraction_response(response)

    def extract_entities(self, text: str) -> list[Entity]:
        prompt = f"""ENTITY EXTRACTION
Extract every explicitly stated named or meaningful entity from the source text.
Do not infer facts. Entity types are open-ended, not restricted to a fixed list.

Source text:
{text}

Return ONLY JSON: {{"entities":[{{"name":"...","type":"...","description":"..."}}]}}"""
        from graphrag.graph.parser import _parse_json_object
        response = self.llm.generate(prompt)
        try:
            return parse_entities(_parse_json_object(response).get("entities", []))
        except ValueError as error:
            repaired = self._repair_json(response, "entities", error)
            return parse_entities(_parse_json_object(repaired).get("entities", []))

    def extract_relationships(self, text: str, entities: list[Entity]) -> list[Relationship]:
        entity_context = "\n".join(f"- {entity.name}" for entity in entities) or "- none"
        self.last_rejections = []
        prompt = f"""RELATIONSHIP EXTRACTION
Extract only relationships explicitly supported by the source text. Sources and targets
must exactly match an entity in the supplied list. Do not invent facts.

Source text:
{text}

Entities:
{entity_context}

Return ONLY JSON: {{"relationships":[{{"source":"...","target":"...","description":"...","weight":1.0}}]}}"""
        response = self.llm.generate(prompt)
        try:
            relationships, self.last_rejections = parse_relationship_response_with_rejections(response, entities)
            return relationships
        except ValueError as error:
            repaired = self._repair_json(response, "relationships", error)
            relationships, self.last_rejections = parse_relationship_response_with_rejections(repaired, entities)
            return relationships

    def _repair_json(self, response: str, expected_key: str, error: ValueError) -> str:
        """Ask the same configured model to fix syntax only; no facts are added."""
        return self.llm.generate(f"""JSON REPAIR
The following model output was rejected because it is not valid JSON or has an
invalid shape. Return ONLY valid JSON with a top-level '{expected_key}' array.
Preserve only facts already present in the original response; do not add facts.

Validation error: {error}
Original response:
{response}
""")
