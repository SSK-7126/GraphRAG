import tempfile
import unittest
from pathlib import Path

from graphrag.graph.builder import GraphBuilder
from graphrag.graph.community import (
    ConnectedComponentsCommunityDetector,
    LeidenCommunityDetector,
)
from graphrag.graph.community_summarizer import CommunitySummarizer
from graphrag.graph.entity import Entity
from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.llm_extractor import LLMGraphExtractor
from graphrag.graph.relationship import Relationship
from graphrag.ingestion.chunker import chunk_text
from graphrag.ingestion.document_pipeline import DocumentPipeline
from graphrag.ingestion.loader import load_text_file
from graphrag.llm.mock import MockLLM
from graphrag.query.global_query import GlobalQueryEngine
from graphrag.query.local import LocalQueryEngine


class GraphRAGTests(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLM()
        self.builder = GraphBuilder(LLMGraphExtractor(self.llm))

    def build_example_graph(self):
        graph = KnowledgeGraph()
        self.builder.add_text(graph, "Alice works at Microsoft.")
        self.builder.add_text(graph, "Microsoft develops Azure.")
        return graph

    def test_loader_and_chunker(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "example.txt"
            file_path.write_text("abcdefghij", encoding="utf-8")
            self.assertEqual(load_text_file(str(file_path)), "abcdefghij")
        self.assertEqual(chunk_text("abcdefghij", chunk_size=4, overlap=1), ["abcd", "defg", "ghij", "j"])

    def test_entity_deduplication_returns_stored_entity(self):
        graph = KnowledgeGraph()
        first = graph.add_entity(Entity("one", "Microsoft", "ORGANIZATION"))
        duplicate = graph.add_entity(Entity("two", "microsoft", "ORGANIZATION"))
        self.assertIs(first, duplicate)
        self.assertEqual(len(graph.entities), 1)

    def test_relationships_are_remapped_after_entity_id_collisions(self):
        graph = self.build_example_graph()
        names_by_id = {entity.id: entity.name for entity in graph.entities.values()}
        edge_names = {
            (names_by_id[relationship.source], names_by_id[relationship.target])
            for relationship in graph.relationships.values()
        }
        self.assertEqual(len(graph.entities), 3)
        self.assertEqual(len(graph.relationships), 2)
        self.assertIn(("Alice", "Microsoft"), edge_names)
        self.assertIn(("Microsoft", "Azure"), edge_names)
        self.assertEqual(graph.validate(), [])

    def test_rejects_dangling_relationships(self):
        graph = KnowledgeGraph()
        with self.assertRaises(ValueError):
            graph.add_relationship(Relationship("edge", "missing", "also_missing"))

    def test_relationship_creation(self):
        graph = KnowledgeGraph()
        source = graph.add_entity(Entity("source", "Alice", "PERSON"))
        target = graph.add_entity(Entity("target", "Microsoft", "ORGANIZATION"))
        relationship = graph.add_relationship(
            Relationship("works_at", source.id, target.id, "Alice works at Microsoft.")
        )
        self.assertIs(graph.get_relationship(relationship.id), relationship)
        self.assertEqual(graph.validate(), [])

    def test_document_pipeline_processes_file(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "example.txt"
            file_path.write_text("Alice works at Microsoft.\n\nMicrosoft develops Azure.", encoding="utf-8")
            graph = DocumentPipeline(self.builder).process_file(str(file_path), KnowledgeGraph())
        self.assertEqual(len(graph.entities), 3)
        self.assertEqual(graph.validate(), [])

    def test_community_detection_and_summarization(self):
        graph = self.build_example_graph()
        communities = ConnectedComponentsCommunityDetector().detect(graph)
        summaries = CommunitySummarizer(self.llm).summarize(graph, communities)
        self.assertEqual(len(communities), 1)
        self.assertEqual(set(communities[0].entity_ids), set(graph.entities))
        self.assertIn("community", summaries[0].summary.lower())

    def test_leiden_detector_uses_safe_fallback_without_optional_packages(self):
        graph = self.build_example_graph()
        communities = LeidenCommunityDetector().detect(graph)
        self.assertEqual(len(communities), 1)

    def test_local_and_global_queries(self):
        graph = self.build_example_graph()
        local = LocalQueryEngine(self.llm).query(graph, "Where does Alice work?")
        summaries = CommunitySummarizer(self.llm).summarize(
            graph, ConnectedComponentsCommunityDetector().detect(graph)
        )
        global_result = GlobalQueryEngine(self.llm).query(summaries, "What is the main theme?")
        self.assertIn("Local answer", local.answer)
        self.assertGreaterEqual(len(local.relationships), 1)
        self.assertIn("Global answer", global_result.answer)
        self.assertEqual(len(global_result.partial_answers), 1)


if __name__ == "__main__":
    unittest.main()
