from dataclasses import dataclass, field

@dataclass
class Relationship:
    id : str
    source:str
    target: str
    description : str =""
    weight : float = 1.0
    metadata: dict = field(default_factory=dict)