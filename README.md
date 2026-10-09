# GraphRAG From Scratch

An intentionally small, educational implementation of the core GraphRAG idea from
Microsoft's *From Local to Global* paper. It converts documents into a knowledge
graph, groups related entities into communities, summarizes those communities, and
answers both focused and high-level questions.

It runs end to end with `MockLLM`, so no API key or paid service is required.

## Architecture

```text
text files
   |
loader -> overlapping chunks -> entity/relationship extraction -> KnowledgeGraph
                                                               |
                                  community detection <--------+
                                     |
                              community summaries
                               /                \
                  local graph query          global map/reduce query
```

The graph is the indexing structure. SentenceTransformer embeddings remain an
optional complementary retrieval mechanism; they do not construct communities or
replace graph extraction.

## Project layout

```text
graphrag/
  ingestion/       Load files and split text into chunks.
  graph/           Entities, edges, validation, extraction, communities, summaries.
  llm/             Swappable MockLLM and OpenAI-compatible client.
  query/           Local-neighbourhood and global map/reduce queries.
  embeddings/      Optional SentenceTransformer embedding interface.
scripts/run_graphrag.py   End-to-end demo.
tests/                    Deterministic standard-library unit tests.
```

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/) are expected.

```bash
uv sync
```

`sentence-transformers` is already declared for optional embedding experiments.
The main graph demo does not download or load an embedding model.

## Run

Use the bundled example with the deterministic mock model:

```bash
uv run python scripts/run_graphrag.py
```

Ingest one or more files and ask a question:

```bash
uv run python scripts/run_graphrag.py data/raw/example.txt --question "How are Alice, Microsoft, and Azure connected?"
```

To use a real OpenAI-compatible endpoint, set an environment variable first; the
key is never stored in code:

```bash
export OPENAI_API_KEY="..."
uv run python scripts/run_graphrag.py --openai --model gpt-4o-mini
```

`OPENAI_BASE_URL` is also supported for compatible endpoints.

Example input (`data/raw/example.txt`):

```text
Alice works at Microsoft.
Microsoft develops Azure.
```

Example mock output:

```text
Ingested 3 entities and 2 relationships.
Communities: 1
Local answer: Alice works at Microsoft, and Microsoft develops Azure.
Global answer: the community evidence has been combined into a high-level answer.
```

## Components

- **Graph construction:** `LLMGraphExtractor` asks an `LLM` for JSON; the parser
  creates `Entity` and `Relationship` objects. `GraphBuilder` deduplicates entity
  names and remaps every edge to the stored entity IDs. `KnowledgeGraph.validate()`
  checks that no edge is dangling.
- **Communities:** `ConnectedComponentsCommunityDetector` is the default,
  dependency-free baseline. It treats every connected component as a community,
  which makes its behavior easy to explain. `LeidenCommunityDetector` is a drop-in
  adapter that uses `igraph` and `leidenalg` when installed, otherwise it falls
  back to the baseline.
- **Summaries:** `CommunitySummarizer` sends the entities and internal
  relationships of each community to the LLM and stores a `CommunitySummary`.
- **Local query:** `LocalQueryEngine` finds entities matching the question,
  collects their one-hop edges and neighbours, then asks the LLM to answer from
  that focused context.
- **Global query:** `GlobalQueryEngine` asks for a short answer from every
  community summary (map), then combines those answers into one response (reduce).

## Vector RAG vs. GraphRAG

Vector RAG finds text chunks similar to the query using embeddings. It is useful
for semantic similarity but does not explicitly model entities, relations, or
community-level themes. GraphRAG first extracts a knowledge graph, which enables
entity neighbourhood retrieval for local questions and community summaries for
global questions. This repository keeps `graphrag/embeddings/` as an optional
hybrid-retrieval extension, not as the core index.

## Test

```bash
uv run python -m unittest discover -s tests -v
```

The suite covers loading, chunking, entity deduplication, remapped relationship
IDs, dangling-edge prevention, document ingestion, communities, summaries, and
both query modes. It uses `MockLLM`, making it fast and deterministic.

## Learning notes and limitations

For a presentation, explain the distinction between **local** (one entity
neighbourhood) and **global** (community map/reduce) questions, and why summaries
let a model reason about a large graph without receiving every edge at once.

This is a learning implementation, not a production GraphRAG system. The mock
extractor only understands the bundled example; real extraction quality depends on
the selected LLM and prompt. The baseline community algorithm is connected
components, not hierarchical Leiden clustering. Graph persistence, provenance,
token budgeting, robust JSON repair, and vector indexing are intentionally left as
small next steps.
