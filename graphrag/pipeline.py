"""Sequential dataset indexing, querying, and benchmark evaluation orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

from graphrag.config import PipelineConfig
from graphrag.datasets import DATASET_NAMES, DatasetDiscoverer, DatasetDiscovery, write_json_atomic
from graphrag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from graphrag.graph.builder import GraphBuilder
from graphrag.graph.community import ConnectedComponentsCommunityDetector
from graphrag.graph.community_summarizer import CommunitySummarizer
from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.llm_extractor import LLMGraphExtractor
from graphrag.ingestion.chunker import chunk_document
from graphrag.ingestion.loader import clean_text, load_text_file
from graphrag.llm.client import LLM
from graphrag.persistence import (
    CheckpointStore, load_graph, load_summaries, save_communities, save_graph, save_summaries,
)
from graphrag.query.global_query import GlobalQueryEngine
from graphrag.query.local import LocalQueryEngine
from graphrag.timing import TimingRecorder
from parallel import ExecutionBackend, SerialBackend
from parallel.base import ExtractionTask


class SequentialDatasetPipeline:
    """The intentional single-process, synchronous GraphRAG execution path."""

    def __init__(self, config: PipelineConfig, llm: LLM, *, backend: ExecutionBackend | None = None,
                 run_id: str | None = None, benchmark_root: Path | None = None,
                 embedding_batch_size: int | None = None):
        self.config = config
        self.llm = llm
        self.discoverer = DatasetDiscoverer(config.raw_dir)
        self.discoveries: dict[str, DatasetDiscovery] | None = None
        self.backend = backend or SerialBackend()
        self.run_id = run_id
        self.benchmark_root = benchmark_root or config.raw_dir.parent.parent / "benchmarks"
        self.embedding_batch_size = embedding_batch_size

    def discover_and_validate(self) -> dict[str, DatasetDiscovery]:
        timer = TimingRecorder(self.run_id, backend=self.backend.name)
        with timer.measure("all", "dataset_discovery", items_processed=1):
            discoveries = self.discoverer.discover()
        with timer.measure("all", "dataset_validation", items_processed=len(discoveries)):
            issues = self.discoverer.validate(discoveries)
        self.discoveries = discoveries
        for dataset, discovery in discoveries.items():
            target = self._dataset_dir(dataset) / "manifest.json"
            payload = discovery.as_manifest(self.config.raw_dir)
            payload["validation_issues"] = issues[dataset]
            if target.exists():
                prior_manifest = json.loads(target.read_text(encoding="utf-8"))
                if "processing_state" in prior_manifest:
                    payload["processing_state"] = prior_manifest["processing_state"]
            write_json_atomic(target, payload)
            self._ensure_output_layout(dataset)
            print(f"DATASET: {dataset} | documents discovered: {len(discovery.documents)}")
            for issue in issues[dataset]:
                print(f"WARNING [{dataset}]: {issue}")
        # Discovery timing is stored independently because it covers all datasets.
        timer.write(self._benchmark_dir("all", timer.run_id))
        return discoveries

    def index(self, selected_datasets: list[str]) -> None:
        discoveries = self.discoveries or self.discover_and_validate()
        for dataset in selected_datasets:
            self._index_dataset(discoveries[dataset])

    def _index_dataset(self, discovery: DatasetDiscovery) -> None:
        dataset = discovery.name
        self._ensure_output_layout(dataset)
        paths = self._paths(dataset)
        timer = TimingRecorder(self.run_id, backend=self.backend.name)
        if self.backend.name == "cuda" and hasattr(self.backend, "cuda_info"):
            timer.set_parallel_metrics(cuda=getattr(self.backend, "cuda_info")())
        checkpoint = CheckpointStore(paths["checkpoint"])
        enable_indexes = self.backend.name != "serial"
        graph = load_graph(paths["graph"], enable_indexes=enable_indexes) if paths["graph"].exists() else KnowledgeGraph(enable_indexes=enable_indexes)
        builder = GraphBuilder(LLMGraphExtractor(self.llm))
        processed_this_run = 0
        failed_this_run = 0
        total_chunks = 0
        total_entities = 0
        total_relationships = 0
        total_documents = len(discovery.documents)
        documents_since_checkpoint = 0

        for position, document in enumerate(discovery.documents, start=1):
            if document.id in checkpoint.completed:
                continue
            print(f"\nDOCUMENT: {position}/{total_documents}\nDocument ID: {document.relative_path}")
            try:
                with timer.measure(dataset, "file_loading", document_id=document.id) as result:
                    raw_text = load_text_file(str(document.path))
                    result["items_processed"] = 1
                with timer.measure(dataset, "text_cleaning", document_id=document.id):
                    text = clean_text(raw_text)
                with timer.measure(dataset, "chunking", document_id=document.id) as result:
                    chunks = chunk_document(
                        text, document.id, chunk_size=self.config.chunk_size,
                        overlap=self.config.chunk_overlap,
                    )
                    result["items_processed"] = len(chunks)
                with timer.measure(dataset, "chunk_persistence", document_id=document.id, items_processed=len(chunks)):
                    filename = hashlib.sha1(document.id.encode("utf-8")).hexdigest() + ".json"
                    write_json_atomic(paths["chunks"] / filename, {
                        "document_id": document.id,
                        "source_relative_path": document.relative_path,
                        "chunks": [
                            {"id": chunk.id, "position": chunk.position, "text": chunk.text}
                            for chunk in chunks
                        ],
                    })
                print(f"STAGE: Chunking | Chunks: {len(chunks)}")
                document_started = perf_counter()
                if self.backend.name == "serial":
                    extracted = []
                    for chunk in chunks:
                        entities = self._extract_entities(timer, dataset, document.id, chunk.id, chunk.text, builder)
                        relationships = self._extract_relationships(timer, dataset, document.id, chunk.id, chunk.text, entities, builder)
                        extracted.append((chunk, entities, relationships, builder.extractor.last_rejections))
                else:
                    task_started = perf_counter()
                    extracted_results = self.backend.extract_chunks(
                        self.llm, [ExtractionTask(document.id, chunk.id, chunk.text) for chunk in chunks], self.config.retry_count
                    )
                    timer.set_parallel_metrics(**self.backend.metrics, task_wait_seconds=perf_counter() - task_started)
                    chunks_by_id = {chunk.id: chunk for chunk in chunks}
                    extracted = [(chunks_by_id[result.task.chunk_id], result.entities,
                                  result.relationships, result.rejections) for result in extracted_results]
                    for result in extracted_results:
                        timer.record_completed_task(dataset, "chunk_extraction", duration_seconds=result.duration_seconds,
                            document_id=document.id, chunk_id=result.task.chunk_id, worker_id=result.worker_id,
                            items_processed=len(result.entities) + len(result.relationships))
                        for usage in result.usage: timer.record_llm_usage(usage, retry=result.attempts > 1)
                # Coordinator merge order is stable regardless of task completion order.
                for chunk, entities, relationships, rejections in sorted(extracted, key=lambda item: item[0].position):
                    checkpoint.record_rejections(document.id, chunk.id, rejections)
                    if self.backend.name == "serial":
                        with timer.measure(dataset, "entity_normalization", document_id=document.id, chunk_id=chunk.id, items_processed=len(entities)):
                            builder.normalize_entities(entities)
                    with timer.measure(dataset, "graph_insertion", document_id=document.id, chunk_id=chunk.id, items_processed=len(entities) + len(relationships)):
                        entity_count, relationship_count = builder.add_extraction(
                            graph, entities, relationships, document_id=document.id, chunk_id=chunk.id,
                            validate=self.backend.name == "serial",
                        )
                    total_chunks += 1
                    total_entities += entity_count
                    total_relationships += relationship_count
                checkpoint.mark_completed(document.id)
                processed_this_run += 1
                print(f"DOCUMENT TOTAL: {perf_counter() - document_started:.3f} sec")
            except Exception as error:
                checkpoint.mark_failed(document.id, error)
                failed_this_run += 1
                print(f"ERROR [{document.relative_path}]: {type(error).__name__}: {error}")
            documents_since_checkpoint += 1
            if documents_since_checkpoint >= self.config.checkpoint_frequency:
                self._persist_checkpoint(dataset, graph, checkpoint, timer)
                documents_since_checkpoint = 0

        # Optimized/parallel paths validate once at the safe coordinator boundary.
        if self.backend.name != "serial":
            with timer.measure(dataset, "graph_validation", items_processed=len(graph.relationships)):
                errors = graph.validate()
                if errors: raise ValueError("Invalid knowledge graph: " + "; ".join(errors))
        self._persist_checkpoint(dataset, graph, checkpoint, timer)
        if failed_this_run:
            print(f"{dataset}: {failed_this_run} document(s) failed; rerun to retry them.")
            timer.write(self._benchmark_dir(dataset, timer.run_id))
            return
        self._build_communities(dataset, graph, timer)
        if self.config.enable_embeddings:
            self._embed_documents(dataset, discovery, timer)
        summary = timer.write(self._benchmark_dir(dataset, timer.run_id))
        print(
            f"\nDataset: {dataset}\nDocuments discovered: {total_documents}\n"
            f"Documents processed this run: {processed_this_run}\nDocuments failed: {failed_this_run}\n"
            f"Total chunks this run: {total_chunks}\nTotal extracted entities: {total_entities}\n"
            f"Total extracted relationships: {total_relationships}\n"
            f"Total execution time: {summary['total_runtime_seconds']:.3f} sec"
        )

    def _extract_entities(self, timer, dataset, document_id, chunk_id, text, builder):
        for attempt in range(self.config.retry_count + 1):
            try:
                with timer.measure(dataset, "entity_extraction", document_id=document_id, chunk_id=chunk_id) as result:
                    entities = builder.extractor.extract_entities(text)
                    result["items_processed"] = len(entities)
                    timer.record_llm_usage(self.llm.last_usage, retry=attempt > 0)
                return entities
            except Exception:
                timer.record_llm_usage(self.llm.last_usage, failed=True, retry=attempt > 0)
                if attempt == self.config.retry_count:
                    raise
        raise AssertionError("unreachable")

    def _extract_relationships(self, timer, dataset, document_id, chunk_id, text, entities, builder):
        for attempt in range(self.config.retry_count + 1):
            try:
                with timer.measure(dataset, "relationship_extraction", document_id=document_id, chunk_id=chunk_id) as result:
                    relationships = builder.extractor.extract_relationships(text, entities)
                    result["items_processed"] = len(relationships)
                    timer.record_llm_usage(self.llm.last_usage, retry=attempt > 0)
                return relationships
            except Exception:
                timer.record_llm_usage(self.llm.last_usage, failed=True, retry=attempt > 0)
                if attempt == self.config.retry_count:
                    raise
        raise AssertionError("unreachable")

    def _persist_checkpoint(self, dataset, graph, checkpoint, timer) -> None:
        paths = self._paths(dataset)
        with timer.measure(dataset, "graph_persistence"):
            save_graph(paths["graph"], graph)
            checkpoint.save()
            manifest_path = paths["root"] / "manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"dataset": dataset}
            )
            manifest["processing_state"] = {
                "completed_document_ids": sorted(checkpoint.completed),
                "failed_documents": checkpoint.failed,
                "rejected_relationships": checkpoint.rejected_relationships,
            }
            write_json_atomic(manifest_path, manifest)

    def _build_communities(self, dataset: str, graph: KnowledgeGraph, timer: TimingRecorder) -> None:
        paths = self._paths(dataset)
        with timer.measure(dataset, "community_detection", items_processed=len(graph.entities)):
            communities = ConnectedComponentsCommunityDetector().detect(graph)
            save_communities(paths["communities"], communities)
        summaries = []
        summarizer = CommunitySummarizer(self.llm)
        for community in communities:
            with timer.measure(dataset, "community_summarization", items_processed=1):
                summaries.append(summarizer.summarize_community(graph, community))
                timer.record_llm_usage(self.llm.last_usage)
        save_summaries(paths["summaries"], summaries)

    def _embed_documents(self, dataset: str, discovery: DatasetDiscovery, timer: TimingRecorder) -> None:
        """Optional embeddings stay separate from the graph and execute sequentially."""
        import numpy as np
        paths = self._paths(dataset)
        device, batch_size = None, self.embedding_batch_size or 32
        if self.backend.name == "cuda":
            cuda_info = getattr(self.backend, "cuda_info")()
            timer.set_parallel_metrics(cuda=cuda_info)
            # SentenceTransformers accepts CPU transparently when CUDA is not
            # present; no fake GPU timings are emitted in that case.
            device = "cuda" if cuda_info.get("available") else "cpu"
            if self.embedding_batch_size is None:
                batch_size = int(cuda_info.get("recommended_embedding_batch_size", batch_size))
        embedder = SentenceTransformerEmbedder(self.config.embedding_model, device=device, batch_size=batch_size)
        texts = []
        records = []
        for document in discovery.documents:
            texts.append(clean_text(load_text_file(str(document.path))))
            records.append({"document_id": document.id, "relative_path": document.relative_path})
        with timer.measure(dataset, "embedding_generation", items_processed=len(texts)):
            vectors = embedder.embed_many(texts, batch_size=batch_size)
        if self.backend.name == "cuda":
            timer.set_parallel_metrics(cuda={**getattr(self.backend, "cuda_info")(), "embedding_timing": embedder.last_timing})
        with timer.measure(dataset, "vector_index_construction", items_processed=len(vectors)):
            paths["embeddings"].mkdir(parents=True, exist_ok=True)
            np.save(paths["embeddings"] / "document_vectors.npy", np.asarray(vectors))
            write_json_atomic(paths["embeddings"] / "document_vectors.json", {"documents": records})

    def query(self, selected_datasets: list[str], question: str, mode: str = "local") -> list[dict]:
        results = []
        for dataset in selected_datasets:
            paths = self._paths(dataset)
            if not paths["graph"].exists():
                raise FileNotFoundError(f"No persisted graph for {dataset}. Run indexing first.")
            timer = TimingRecorder(self.run_id, backend=self.backend.name)
            with timer.measure(dataset, "query_parsing"):
                normalized_question = question.strip()
                if not normalized_question:
                    raise ValueError("Question must not be empty.")
            graph = load_graph(paths["graph"])
            with timer.measure(dataset, "query_total"):
                if mode == "local":
                    engine = LocalQueryEngine(self.llm)
                    with timer.measure(dataset, "query_retrieval"):
                        entity_ids, relationships = engine.retrieve(graph, normalized_question)
                    with timer.measure(dataset, "query_llm_generation"):
                        answer = engine.answer(graph, normalized_question, entity_ids, relationships)
                        timer.record_llm_usage(self.llm.last_usage)
                else:
                    if not paths["summaries"].exists():
                        raise FileNotFoundError(f"No community summaries for {dataset}. Run indexing first.")
                    with timer.measure(dataset, "query_retrieval"):
                        summaries = load_summaries(paths["summaries"])
                    with timer.measure(dataset, "query_llm_generation"):
                        answer = GlobalQueryEngine(self.llm).query(summaries, normalized_question).answer
                        timer.record_llm_usage(self.llm.last_usage)
            payload = {"dataset": dataset, "question": normalized_question, "mode": mode, "answer": answer}
            results.append(payload)
            query_dir = paths["queries"]
            query_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(query_dir / f"{timer.run_id}.json", payload)
            timer.write(self._benchmark_dir(dataset, timer.run_id))
        return results

    def evaluate(self, selected_datasets: list[str], mode: str = "local") -> None:
        discoveries = self.discoveries or self.discover_and_validate()
        for dataset in selected_datasets:
            for question_file in discoveries[dataset].question_files:
                with question_file.path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                question_column = next((key for key in rows[0] if key.casefold() in {"question", "question_text"}), None) if rows else None
                if question_column is None:
                    raise ValueError(f"No question column in {question_file.path}")
                output_rows = []
                timer = TimingRecorder(self.run_id, backend=self.backend.name)
                for row in rows:
                    with timer.measure(dataset, "evaluation_question", document_id=row.get("question_id")):
                        result = self.query([dataset], row[question_column], mode)[0]
                    output_rows.append({**row, "generated_answer": result["answer"], "reference_answer": row.get("answer", ""), "error": ""})
                output = self._paths(dataset)["evaluation"] / f"{question_file.kind}_results.csv"
                with output.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys() if output_rows else ["generated_answer"])
                    writer.writeheader()
                    writer.writerows(output_rows)
                timer.write(self._benchmark_dir(dataset, timer.run_id))

    def _dataset_dir(self, dataset: str) -> Path:
        return self.config.processed_dir / dataset

    def _paths(self, dataset: str) -> dict[str, Path]:
        root = self._dataset_dir(dataset)
        return {
            "root": root, "chunks": root / "chunks", "graph": root / "graph" / "knowledge_graph.json",
            "communities": root / "communities" / "communities.json",
            "summaries": root / "communities" / "summaries.json",
            "checkpoint": root / "checkpoints" / "checkpoint.json",
            "timings": root / "timings", "embeddings": root / "embeddings",
            "evaluation": root / "evaluation", "queries": root / "queries",
        }

    def _ensure_output_layout(self, dataset: str) -> None:
        paths = self._paths(dataset)
        for name in ("chunks", "graph", "communities", "embeddings", "checkpoints", "timings", "evaluation", "queries"):
            (paths["root"] / name).mkdir(parents=True, exist_ok=True)

    def _benchmark_dir(self, dataset: str, run_id: str) -> Path:
        """A unique output directory prevents records from distinct runs mixing."""
        return self.benchmark_root / self.backend.name / run_id / dataset


def normalize_dataset_selection(values: list[str]) -> list[str]:
    selection = list(DATASET_NAMES) if not values or "all" in values else values
    invalid = sorted(set(selection) - set(DATASET_NAMES))
    if invalid:
        raise ValueError(f"Unknown dataset selection: {', '.join(invalid)}")
    return selection
