from graphrag.llm.client import LLM
from graphrag.graph.parser import parse_extraction_response
from graphrag.graph.extractor import GraphExtractor
from graphrag.graph.entity import Entity
from graphrag.graph.relationship import Relationship

class LLMGraphExtractor(GraphExtractor):
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, text: str) -> tuple[list[Entity], list[Relationship]]:
        
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
