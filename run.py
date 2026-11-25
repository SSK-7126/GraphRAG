"""Unified reproducible GraphRAG benchmark CLI."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path

from graphrag.config import PipelineConfig
from graphrag.llm.mock import MockLLM
from graphrag.llm.ollama import OllamaLLM
from graphrag.pipeline import SequentialDatasetPipeline, normalize_dataset_selection
from parallel import BACKENDS, create_backend

def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except ImportError:
        return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG with a measured execution backend.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="serial")
    parser.add_argument("--command", choices=("discover", "index", "query", "evaluate"), default="index")
    parser.add_argument("--dataset", action="append", choices=("hotpotqa", "kevin_scott", "msft", "all"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-concurrent-llm-calls", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--embedding-batch-size", type=int, help="CUDA chooses a safe batch size when omitted.")
    parser.add_argument("--enable-embeddings", action="store_true", help="Generate batched embeddings; required for CUDA acceleration.")
    parser.add_argument("--hardware", action="store_true", help="Print CUDA/Ray/Dask availability and exit.")
    parser.add_argument("--run-id")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--raw-dir", type=Path); parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--question"); parser.add_argument("--mode", choices=("local", "global"), default="local")
    parser.add_argument("--llm", choices=("ollama", "mock"), default="ollama", help="Mock is useful for deterministic tests only.")
    args = parser.parse_args()
    if args.workers < 1 or args.repeat < 1 or args.warmup_runs < 0: parser.error("workers/repeat must be positive and warmup-runs non-negative")
    if args.embedding_batch_size is not None and args.embedding_batch_size < 1: parser.error("--embedding-batch-size must be positive")
    config = PipelineConfig.from_environment()
    changes = {}
    for name, value in (("raw_dir", args.raw_dir), ("processed_dir", args.processed_dir), ("chunk_size", args.chunk_size), ("chunk_overlap", args.chunk_overlap)):
        if value is not None: changes[name] = value.expanduser().resolve() if isinstance(value, Path) else value
    if args.enable_embeddings: changes["enable_embeddings"] = True
    if changes: config = replace(config, **changes); config.validate()
    if args.hardware:
        backend = create_backend("cuda")
        print({"cuda": backend.cuda_info(), "ray": _module_version("ray"), "dask": _module_version("dask")})
        return
    selection = normalize_dataset_selection(args.dataset or [])
    total = args.warmup_runs + args.repeat
    for index in range(total):
        backend = create_backend(args.backend, args.workers, args.max_concurrent_llm_calls)
        llm = MockLLM() if args.llm == "mock" else OllamaLLM(config.ollama_model, config.ollama_base_url)
        suffix = f"-{index + 1}" if total > 1 else ""
        run_id = f"{args.run_id}{suffix}" if args.run_id else None
        pipeline = SequentialDatasetPipeline(config, llm, backend=backend, run_id=run_id,
            embedding_batch_size=args.embedding_batch_size)
        if args.command == "discover": pipeline.discover_and_validate()
        elif args.command == "index": pipeline.index(selection)
        elif args.command == "query":
            if not args.question: parser.error("--question is required for query")
            for result in pipeline.query(selection, args.question, args.mode): print(result["answer"])
        else: pipeline.evaluate(selection, args.mode)
        if index < args.warmup_runs: print(f"Completed warm-up {index + 1}/{args.warmup_runs}; result remains recorded but excluded from comparisons.")

if __name__ == "__main__": main()
