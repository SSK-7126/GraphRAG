"""Common, deterministic worker protocol for GraphRAG extraction backends."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from graphrag.graph.builder import GraphBuilder
from graphrag.graph.llm_extractor import LLMGraphExtractor
from graphrag.llm.client import LLM

@dataclass(frozen=True)
class ExtractionTask:
    document_id: str
    chunk_id: str
    text: str

@dataclass
class ExtractionResult:
    task: ExtractionTask
    entities: list
    relationships: list
    rejections: list[str]
    duration_seconds: float
    worker_id: str
    attempts: int
    usage: list[dict | None]

class BackendUnavailableError(RuntimeError):
    pass

class ExecutionBackend:
    name = "serial"
    def __init__(self, workers: int = 1, *, max_concurrent_llm_calls: int | None = None):
        if workers < 1: raise ValueError("workers must be at least 1")
        self.workers = workers
        self.max_concurrent_llm_calls = max_concurrent_llm_calls or workers
        self.metrics: dict[str, Any] = {"worker_count": workers, "tasks": 0, "failed_tasks": 0}
    def extract_chunks(self, llm: LLM, tasks: list[ExtractionTask], retry_count: int) -> list[ExtractionResult]:
        raise NotImplementedError

def llm_spec(llm: LLM) -> dict[str, Any]:
    """Only known clients are portable across processes/cluster workers."""
    return llm.worker_spec()

def llm_from_spec(spec: dict[str, Any]) -> LLM:
    kind = spec["kind"]
    if kind == "mock":
        from graphrag.llm.mock import MockLLM; return MockLLM()
    if kind == "ollama":
        from graphrag.llm.ollama import OllamaLLM; return OllamaLLM(spec["model"], spec["base_url"])
    if kind == "openai_compatible":
        from graphrag.llm.openai import OpenAICompatibleLLM
        return OpenAICompatibleLLM(spec["model"], api_key=spec.get("api_key"), base_url=spec["base_url"])
    raise ValueError(f"Unsupported portable LLM worker spec: {kind!r}")

def execute_extraction_task(spec: dict[str, Any], task: ExtractionTask, retry_count: int, worker_id: str = "worker") -> ExtractionResult:
    """Top-level function so multiprocessing, Ray, and Dask can serialize it."""
    started, usages = perf_counter(), []
    llm = llm_from_spec(spec)
    extractor = LLMGraphExtractor(llm)
    for attempt in range(retry_count + 1):
        try:
            entities = extractor.extract_entities(task.text); usages.append(llm.last_usage)
            relationships = extractor.extract_relationships(task.text, entities); usages.append(llm.last_usage)
            GraphBuilder.normalize_entities(entities)
            return ExtractionResult(task, entities, relationships, extractor.last_rejections,
                perf_counter() - started, worker_id, attempt + 1, usages)
        except Exception:
            usages.append(llm.last_usage)
            if attempt == retry_count: raise
    raise AssertionError("unreachable")

class SerialBackend(ExecutionBackend):
    name = "serial"
    def extract_chunks(self, llm: LLM, tasks: list[ExtractionTask], retry_count: int) -> list[ExtractionResult]:
        spec = llm_spec(llm); self.metrics["tasks"] = len(tasks)
        return [execute_extraction_task(spec, task, retry_count, "main") for task in tasks]

class OptimizedSerialBackend(SerialBackend):
    name = "optimized_serial"
