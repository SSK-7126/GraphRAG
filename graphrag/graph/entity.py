from dataclasses import dataclass, field


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    description: str = ""
    metadata: dict = field(default_factory=dict)