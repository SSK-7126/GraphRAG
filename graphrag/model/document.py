from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    text: str
    source: str

@dataclass
class Chunk:
    id: str
    text: str
    document_id: str
    position: int
    metadata: dict = field(default_factory=dict)