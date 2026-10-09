from graphrag.graph.builder import GraphBuilder
from graphrag.graph.graph import KnowledgeGraph

class IngestionPipeline:
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder

    def process_chunks(self, chunks: list[str], graph: KnowledgeGraph) -> KnowledgeGraph:
        for chunk in chunks:
            self.graph_builder.add_text(graph, chunk)
        return graph
