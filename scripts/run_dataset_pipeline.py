"""Command-line interface for the sequential, Ollama-only dataset workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag.config import PipelineConfig
from graphrag.llm.ollama import OllamaLLM
from graphrag.pipeline import SequentialDatasetPipeline, normalize_dataset_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sequential GraphRAG dataset discovery, indexing, querying, or evaluation."
    )
    parser.add_argument("command", choices=("discover", "index", "query", "evaluate"))
    parser.add_argument(
        "--dataset", action="append", choices=("hotpotqa", "kevin_scott", "msft", "all"),
        help="Repeat for multiple datasets; defaults to all.",
    )
    parser.add_argument("--question", help="Question for the query command.")
    parser.add_argument("--mode", choices=("local", "global"), default="local")
    parser.add_argument("--raw-dir", type=Path, help="Override GRAPHRAG_RAW_DIR for this command.")
    parser.add_argument("--processed-dir", type=Path, help="Override GRAPHRAG_PROCESSED_DIR for this command.")
    args = parser.parse_args()

    config = PipelineConfig.from_environment()
    if args.raw_dir:
        config = replace(config, raw_dir=args.raw_dir.expanduser().resolve())
    if args.processed_dir:
        config = replace(config, processed_dir=args.processed_dir.expanduser().resolve())
    llm = OllamaLLM(config.ollama_model, config.ollama_base_url)
    pipeline = SequentialDatasetPipeline(config, llm)
    selection = normalize_dataset_selection(args.dataset or [])

    if args.command == "discover":
        pipeline.discover_and_validate()
    elif args.command == "index":
        pipeline.index(selection)
    elif args.command == "query":
        if not args.question:
            parser.error("--question is required for query")
        for result in pipeline.query(selection, args.question, args.mode):
            print(f"\nDATASET: {result['dataset']}\nANSWER: {result['answer']}")
    else:
        pipeline.evaluate(selection, args.mode)


if __name__ == "__main__":
    main()
