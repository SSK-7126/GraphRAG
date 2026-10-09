"""Run the complete educational GraphRAG pipeline over one or more text files."""

import argparse
from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag.graph.builder import GraphBuilder
from graphrag.graph.community import ConnectedComponentsCommunityDetector
from graphrag.graph.community_summarizer import CommunitySummarizer
from graphrag.graph.graph import KnowledgeGraph
from graphrag.graph.llm_extractor import LLMGraphExtractor
from graphrag.ingestion.document_pipeline import DocumentPipeline
from graphrag.llm.mock import MockLLM
from graphrag.llm.openai import OpenAICompatibleLLM
from graphrag.query.global_query import GlobalQueryEngine
from graphrag.query.local import LocalQueryEngine


def build_llm(use_openai: bool, model: str):
    return OpenAICompatibleLLM(model=model) if use_openai else MockLLM()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simplified GraphRAG pipeline.")
    parser.add_argument(
        "files",
        nargs="*",
        default=["data/raw/example.txt"],
        help="Text files to ingest (default: data/raw/example.txt).",
    )
    parser.add_argument("--question", default="What is the relationship between Alice and Azure?")
    parser.add_argument("--openai", action="store_true", help="Use OPENAI_API_KEY instead of MockLLM.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for --openai.")
    args = parser.parse_args()

    llm = build_llm(args.openai, args.model)
    graph = KnowledgeGraph()
    document_pipeline = DocumentPipeline(GraphBuilder(LLMGraphExtractor(llm)))
    for file_path in args.files:
        document_pipeline.process_file(file_path, graph)

    communities = ConnectedComponentsCommunityDetector().detect(graph)
    summaries = CommunitySummarizer(llm).summarize(graph, communities)
    local_result = LocalQueryEngine(llm).query(graph, args.question)
    global_result = GlobalQueryEngine(llm).query(summaries, args.question)

    print(f"Ingested {len(graph.entities)} entities and {len(graph.relationships)} relationships.")
    print(f"Communities: {len(communities)}")
    for summary in summaries:
        names = ", ".join(graph.entities[entity_id].name for entity_id in summary.entity_ids)
        print(f"- {summary.community_id} ({names}): {summary.summary}")
    print(f"\nLocal answer: {local_result.answer}")
    print(f"Global answer: {global_result.answer}")


if __name__ == "__main__":
    main()
