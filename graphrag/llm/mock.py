from graphrag.llm.client import LLM


class MockLLM(LLM):

    def generate(self, prompt: str) -> str:
        if "GLOBAL MAP" in prompt:
            return "This community contributes evidence relevant to the global question."

        if "GLOBAL REDUCE" in prompt:
            return "Global answer: the community evidence has been combined into a high-level answer."

        if "LOCAL QUERY" in prompt:
            return "Local answer: Alice works at Microsoft, and Microsoft develops Azure."

        if "COMMUNITY SUMMARY" in prompt:
            return "This community describes the entities and relationships supplied in the graph context."

        if "Alice works at Microsoft." in prompt and "Microsoft develops Azure." in prompt:
            return """
            {
                "entities": [
                    {"name": "Alice", "type": "PERSON", "description": "A person who works at Microsoft."},
                    {"name": "Microsoft", "type": "ORGANIZATION", "description": "A technology company."},
                    {"name": "Azure", "type": "PRODUCT", "description": "A cloud computing service developed by Microsoft."}
                ],
                "relationships": [
                    {"source": "Alice", "target": "Microsoft", "description": "Alice works at Microsoft."},
                    {"source": "Microsoft", "target": "Azure", "description": "Microsoft develops Azure."}
                ]
            }
            """

        if "Alice works at Microsoft." in prompt:
            return """
        {
            "entities": [
                {
                    "name": "Alice",
                    "type": "PERSON",
                    "description": "A person who works at Microsoft."
                },
                {
                    "name": "Microsoft",
                    "type": "ORGANIZATION",
                    "description": "A technology company."
                }
            ],
            "relationships": [
                {
                    "source": "Alice",
                    "target": "Microsoft",
                    "description": "Alice works at Microsoft."
                }
            ]
        }
        """

        if "Microsoft develops Azure." in prompt:
            return """
            {
                "entities": [
                    {
                        "name": "Microsoft",
                        "type": "ORGANIZATION",
                        "description": "A technology company."
                    },
                    {
                        "name": "Azure",
                        "type": "PRODUCT",
                        "description": "A cloud computing service developed by Microsoft."
                    }
                ],
                "relationships": [
                    {
                        "source": "Microsoft",
                        "target": "Azure",
                        "description": "Microsoft develops Azure."
                    }
                ]
            }
            """

        return """
        {
            "entities": [],
            "relationships": []
        }
        """
