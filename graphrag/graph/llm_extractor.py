from graphrag.llm.client import LLM
from graphrag.graph.parser import parse_extraction_response
from graphrag.graph.extractor import GraphExtractor
from graphrag.graph.entity import Entity
from graphrag.graph.relationship import Relationship

class LLMGraphExtractor(GraphExtractor):
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, text: str) -> tuple[list[Entity], list[Relationship]]:
        prompt = f"""Extract entities and relationships from the following text.
Text: {text}

Return only JSON in this format:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
 "relationships": [{{"source": "entity name", "target": "entity name", "description": "..."}}]}}."""

        response = self.llm.generate(prompt)
        return parse_extraction_response(response)
