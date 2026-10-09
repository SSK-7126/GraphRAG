from graphrag.graph.entity import Entity
from graphrag.graph.relationship import Relationship
from graphrag.graph.extractor import GraphExtractor

class MockGraphExtractor(GraphExtractor):
    def extract(self,text:str)->tuple[list[Entity],list[Relationship]]:
        entities=[
            Entity(id="entity_001",name="Alice",entity_type="Person",description="A software engineer."),
            Entity(id="entity_002",name="Microsoft",entity_type="Company",description="A technology company.")           
        ]

        relationships=[
            Relationship(id="rel_001",source="entity_001",target="entity_002",description="Alice works at Microsoft.")
        ]
        return entities, relationships
