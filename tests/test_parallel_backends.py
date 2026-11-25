import tempfile
import unittest
from pathlib import Path

from graphrag.graph.builder import GraphBuilder
from graphrag.graph.entity import Entity
from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.relationship import Relationship
from graphrag.llm.mock import MockLLM
from graphrag.timing import TimingRecorder
from parallel import create_backend
from parallel.base import ExtractionTask


class ParallelBackendTests(unittest.TestCase):
    def test_asyncio_returns_isolated_extractions_in_input_order(self):
        tasks = [
            ExtractionTask("doc", "doc:chunk:00001", "Alice works at Microsoft."),
            ExtractionTask("doc", "doc:chunk:00002", "Microsoft develops Azure."),
        ]
        results = create_backend("asyncio", workers=2).extract_chunks(MockLLM(), tasks, 0)
        self.assertEqual([result.task.chunk_id for result in results], [task.chunk_id for task in tasks])
        graph = KnowledgeGraph()
        builder = GraphBuilder(MockLLM())  # constructor argument is unused by add_extraction
        for result in results:
            builder.add_extraction(graph, result.entities, result.relationships,
                                   document_id=result.task.document_id, chunk_id=result.task.chunk_id)
        self.assertEqual(graph.validate(), [])

    def test_indexed_relationship_duplicate_keeps_provenance(self):
        graph = KnowledgeGraph()
        source = graph.add_entity(Entity("a", "Alice", "person"))
        target = graph.add_entity(Entity("m", "Microsoft", "company"))
        first = graph.add_relationship(Relationship("r1", source.id, target.id, "works at", metadata={"source_chunks": ["c1"]}))
        second = graph.add_relationship(Relationship("r2", source.id, target.id, "works at", metadata={"source_chunks": ["c2"]}))
        self.assertIs(first, second)
        self.assertEqual(first.metadata["source_chunks"], ["c1", "c2"])

    def test_timing_outputs_chunk_file_and_wall_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            timer = TimingRecorder("run", backend="asyncio")
            with timer.measure("fixture", "entity_extraction", document_id="d", chunk_id="c"):
                pass
            summary = timer.write(Path(directory))
            self.assertIn("wall_clock_runtime_seconds", summary)
            self.assertTrue((Path(directory) / "chunk_timings.csv").exists())


if __name__ == "__main__":
    unittest.main()
