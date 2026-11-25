import tempfile
import unittest
from pathlib import Path

from graphrag.datasets import DatasetDiscoverer
from graphrag.graph.entity import Entity
from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.relationship import Relationship
from graphrag.persistence import CheckpointStore, load_graph, save_graph
from graphrag.timing import TimingRecorder


class DatasetSupportTests(unittest.TestCase):
    def test_discovery_classifies_by_content_and_csv_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "input").mkdir()
            (raw / "input" / "test_1.txt").write_text("Ada Lovelace was a mathematician.", encoding="utf-8")
            (raw / "HotPotQA Questions.csv").write_text("question_text,question_id\nWho was Ada?,1\n", encoding="utf-8")
            discovery = DatasetDiscoverer(raw).discover()
            self.assertEqual(len(discovery["hotpotqa"].documents), 1)
            self.assertEqual(discovery["hotpotqa"].question_files[0].kind, "benchmark")

    def test_graph_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = KnowledgeGraph()
            alice = graph.add_entity(Entity("alice", "Alice", "person", metadata={"source_documents": ["d1"]}))
            microsoft = graph.add_entity(Entity("ms", "Microsoft", "company"))
            graph.add_relationship(Relationship("works", alice.id, microsoft.id, "works at"))
            graph_path = root / "graph.json"
            save_graph(graph_path, graph)
            self.assertEqual(len(load_graph(graph_path).relationships), 1)
            checkpoint = CheckpointStore(root / "checkpoint.json")
            checkpoint.mark_completed("d1")
            checkpoint.mark_failed("d2", ValueError("bad response"))
            checkpoint.save()
            loaded = CheckpointStore(root / "checkpoint.json")
            self.assertEqual(loaded.completed, {"d1"})
            self.assertIn("d2", loaded.failed)

    def test_timing_has_required_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            timer = TimingRecorder("test-run")
            with timer.measure("hotpotqa", "chunking", document_id="doc", items_processed=2):
                pass
            timer.write(Path(directory))
            header = (Path(directory) / "stage_timings.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header.split(","), [
                "run_id", "dataset", "document_id", "chunk_id", "stage", "start_timestamp",
                "end_timestamp", "duration_seconds", "items_processed", "success_count", "failure_count",
            ])


if __name__ == "__main__":
    unittest.main()
