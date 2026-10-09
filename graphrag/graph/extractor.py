from abc import ABC, abstractmethod
from graphrag.graph.entity import Entity
from graphrag.graph.relationship import Relationship

class GraphExtractor(ABC):
    @abstractmethod
    def extract(self,text:str)->tuple[list[Entity],list[Relationship]]:
        pass