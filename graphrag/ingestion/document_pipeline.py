from graphrag.ingestion.loader import load_text_file
from graphrag.ingestion.chunker import chunk_text
from graphrag.graph.builder import GraphBuilder
from graphrag.graph.graph import KnowledgeGraph

class DocumentPipeline:
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder

    def process_file(self, file_path: str, graph: KnowledgeGraph) -> KnowledgeGraph:
        text = load_text_file(file_path)
        chunks = chunk_text(text)

        for chunk in chunks:
            self.graph_builder.add_text(graph, chunk)
        return graph
