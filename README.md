# GraphRAG From Scratch

An intentionally small, educational implementation of the core GraphRAG idea from
Microsoft's *From Local to Global* paper. It converts documents into a knowledge
graph, groups related entities into communities, summarizes those communities, and
answers both focused and high-level questions.

Production dataset commands use a local Ollama model only; they never fall back
to an online provider.

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
  llm/             LLM abstraction, Ollama client, and deterministic test mock.
  query/           Local-neighbourhood and global map/reduce queries.
  embeddings/      Optional SentenceTransformer embedding interface.
scripts/run_dataset_pipeline.py  Sequential discovery/index/query/evaluation CLI.
scripts/run_graphrag.py          Small Ollama demo.
tests/                    Deterministic standard-library unit tests.
```

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/) are expected.

```bash
uv sync
```

`sentence-transformers` is already declared for optional embedding experiments.
The main graph demo does not download or load an embedding model.

## Dataset pipeline

The extracted datasets in `data/raw` are discovered as HotPotQA (5,491 text
files), Kevin Scott Podcasts (1,669), and Microsoft earnings calls (41). All
generated data goes beneath `data/processed` by default, never into `raw`.
Copy the values needed from `.env.example` into your environment; no dotenv
package is required.

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
# Optional external locations:
$env:GRAPHRAG_RAW_DIR = "C:\Users\sreek\Sreekar\folder\raw"
$env:GRAPHRAG_PROCESSED_DIR = "C:\Users\sreek\Sreekar\folder\processed"
```

Start Ollama and pull the chosen model yourself; the code neither starts the
server nor downloads models. First inspect the no-LLM discovery report:

```powershell
uv run python scripts/run_dataset_pipeline.py discover
```

Index one collection or all collections in intentionally sequential order:

```powershell
uv run python scripts/run_dataset_pipeline.py index --dataset hotpotqa
uv run python scripts/run_dataset_pipeline.py index --dataset all
```

Resume is automatic: `processed/<dataset>/checkpoints/checkpoint.json` records
only completed documents; failures are retained and retried on the next run.
Each dataset receives independent graph, community, checkpoint, timing,
evaluation, embedding, and query directories. Inspect
`processed/<dataset>/timings/summary.json` for aggregate timings and the adjacent
CSVs for per-stage, per-document, query, and evaluation timing details.

Query one or every indexed dataset, with each result labelled:

```powershell
uv run python scripts/run_dataset_pipeline.py query --dataset msft --question "What affected revenue?"
uv run python scripts/run_dataset_pipeline.py query --dataset all --question "What is the main theme?" --mode global
```

Run the discovered question CSVs against their correct dataset graphs:

```powershell
uv run python scripts/run_dataset_pipeline.py evaluate --dataset msft
```

The supplied CSVs contain `question_text` and `question_id`, but no reference
answer column. The evaluator saves generated answers with an empty reference
field and keeps MSFT single- and multi-transcript results separate. Embeddings
remain disabled by default; enable `GRAPHRAG_ENABLE_EMBEDDINGS=true` only when
you want to load the configured Sentence Transformer during indexing.

## Small demo

Use the bundled example with Ollama:

```bash
uv run python scripts/run_graphrag.py
```

Ingest one or more files and ask a question:

```bash
uv run python scripts/run_graphrag.py data/raw/example.txt --question "How are Alice, Microsoft, and Azure connected?"
```

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
both query modes. It uses `MockLLM`, making it fast and deterministic; production
commands always use Ollama.

## Learning notes and limitations

For a presentation, explain the distinction between **local** (one entity
neighbourhood) and **global** (community map/reduce) questions, and why summaries
let a model reason about a large graph without receiving every edge at once.

This is a deliberately sequential implementation so timing data can guide later
parallelisation. Extraction quality remains model- and prompt-dependent. The
baseline community algorithm is connected components rather than hierarchical
Leiden clustering. Token counts are reported only when Ollama supplies them; no
metric is fabricated.
