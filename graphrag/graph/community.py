"""Community models and replaceable graph community detection strategies."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from graphrag.graph.graph import KnowledgeGraph


@dataclass
class Community:
    id: str
    entity_ids: list[str]
    metadata: dict = field(default_factory=dict)


class CommunityDetector(ABC):
    @abstractmethod
    def detect(self, graph: KnowledgeGraph) -> list[Community]:
        """Partition a knowledge graph into groups of related entities."""


class ConnectedComponentsCommunityDetector(CommunityDetector):
    """A dependency-free baseline: each connected graph component is a community."""

    def detect(self, graph: KnowledgeGraph) -> list[Community]:
        adjacency = {entity_id: set() for entity_id in graph.entities}
        for relationship in graph.relationships.values():
            adjacency[relationship.source].add(relationship.target)
            adjacency[relationship.target].add(relationship.source)

        communities = []
        visited = set()
        for entity_id in sorted(adjacency):
            if entity_id in visited:
                continue
            queue = deque([entity_id])
            component = []
            visited.add(entity_id)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in sorted(adjacency[current]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            communities.append(
                Community(id=f"community_{len(communities) + 1}", entity_ids=component)
            )
        return communities


class LeidenCommunityDetector(CommunityDetector):
    """Optional Leiden detector with a clear fallback when its packages are absent."""

    def __init__(self, fallback_to_connected_components: bool = True):
        self.fallback_to_connected_components = fallback_to_connected_components

    def detect(self, graph: KnowledgeGraph) -> list[Community]:
        try:
            import igraph as ig
            import leidenalg
        except ImportError as error:
            if self.fallback_to_connected_components:
                return ConnectedComponentsCommunityDetector().detect(graph)
            raise RuntimeError(
                "Leiden requires optional packages 'igraph' and 'leidenalg'."
            ) from error

        entity_ids = list(graph.entities)
        index_by_id = {entity_id: index for index, entity_id in enumerate(entity_ids)}
        network = ig.Graph(n=len(entity_ids), directed=False)
        edges = [
            (index_by_id[relationship.source], index_by_id[relationship.target])
            for relationship in graph.relationships.values()
        ]
        network.add_edges(edges)
        partition = leidenalg.find_partition(network, leidenalg.ModularityVertexPartition)
        return [
            Community(
                id=f"community_{index + 1}",
                entity_ids=[entity_ids[vertex] for vertex in members],
                metadata={"algorithm": "leiden"},
            )
            for index, members in enumerate(partition)
        ]
