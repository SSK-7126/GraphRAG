"""Atomic persistence for graphs, communities, summaries, and run checkpoints."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from graphrag.graph.community import Community
from graphrag.graph.community_summarizer import CommunitySummary
from graphrag.graph.entity import Entity
from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.relationship import Relationship


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def save_graph(path: Path, graph: KnowledgeGraph) -> None:
    _write_json_atomic(path, {
        "entities": [asdict(entity) for entity in graph.entities.values()],
        "relationships": [asdict(relationship) for relationship in graph.relationships.values()],
    })


def load_graph(path: Path, *, enable_indexes: bool = True) -> KnowledgeGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = KnowledgeGraph(enable_indexes=enable_indexes)
    for item in data.get("entities", []):
        graph.entities[item["id"]] = Entity(**item)
    for item in data.get("relationships", []):
        relationship = Relationship(**item)
        if relationship.source not in graph.entities or relationship.target not in graph.entities:
            raise ValueError(f"Persisted graph has dangling relationship {relationship.id}")
        graph.relationships[relationship.id] = relationship
    graph.rebuild_indexes()
    errors = graph.validate()
    if errors:
        raise ValueError("Persisted graph is invalid: " + "; ".join(errors))
    return graph


def save_communities(path: Path, communities: list[Community]) -> None:
    _write_json_atomic(path, [asdict(community) for community in communities])


def load_communities(path: Path) -> list[Community]:
    return [Community(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def save_summaries(path: Path, summaries: list[CommunitySummary]) -> None:
    _write_json_atomic(path, [asdict(summary) for summary in summaries])


def load_summaries(path: Path) -> list[CommunitySummary]:
    return [CommunitySummary(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


class CheckpointStore:
    """Dataset-local checkpoint state; failures never enter ``completed``."""

    def __init__(self, path: Path):
        self.path = path
        self.completed: set[str] = set()
        self.failed: dict[str, str] = {}
        self.rejected_relationships: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.completed = set(data.get("completed_document_ids", []))
        self.failed = dict(data.get("failed_documents", {}))
        self.rejected_relationships = dict(data.get("rejected_relationships", {}))

    def mark_completed(self, document_id: str) -> None:
        self.completed.add(document_id)
        self.failed.pop(document_id, None)

    def mark_failed(self, document_id: str, error: Exception) -> None:
        self.failed[document_id] = f"{type(error).__name__}: {error}"

    def record_rejections(self, document_id: str, chunk_id: str, messages: list[str]) -> None:
        if messages:
            self.rejected_relationships[f"{document_id}|{chunk_id}"] = messages

    def save(self) -> None:
        _write_json_atomic(self.path, {
            "completed_document_ids": sorted(self.completed),
            "failed_documents": self.failed,
            "rejected_relationships": self.rejected_relationships,
        })
