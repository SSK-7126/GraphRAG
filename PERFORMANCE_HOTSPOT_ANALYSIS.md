# Performance Hotspot Analysis

This analysis is based on source inspection and the measured mock fixture run in
`benchmarks/serial/baseline-mock/msft`. The fixture uses a deterministic local
mock, so its timing must not be generalized to an LLM-backed production run.
No performance percentage below is an invented measurement.

| Rank | Location | Current cost / complexity | Bound | Change implemented | Suitable backend / limitation |
|---:|---|---|---|---|---|
| 1 | `pipeline._extract_entities`, `_extract_relationships` | One or two remote LLM calls per chunk; `O(chunks * LLM latency)` | I/O / remote inference | Isolated chunk workers and deterministic coordinator merge | asyncio is appropriate for a rate-limited server; Ray/Dask for remote workers. The model server remains the bottleneck. |
| 2 | `KnowledgeGraph.add_entity` | Original linear normalized-name scan per entity, `O(E)` | CPU | normalized-name index; expected lookup `O(1)` | optimized serial; merging stays coordinator-only. |
| 3 | `KnowledgeGraph.add_relationship` | Original complete relationship scan, `O(R)` | CPU | endpoint/description/chunk provenance indexes; expected `O(1)` lookup | optimized serial and coordinator only; semantics retained. |
| 4 | `GraphBuilder.add_extraction` | Full `validate()` after every chunk: cumulative `O(chunks * R)` | CPU | safe endpoint checks during insert plus full validation at checkpoint/final boundary | optimized serial / coordinator. Baseline still validates per chunk. |
| 5 | `CommunitySummarizer.summarize_community` | Each community scans every edge, up to `O(communities * R)` | CPU + I/O | indexed incident edges then internal-edge filter | asyncio/Ray/Dask can parallelize LLM summaries once clients are isolated. |
| 6 | `LocalQueryEngine._find_relevant_entities` | Rebuilds strings and sorts all matches, `O(E * query_terms + M log M)` | CPU | cached searchable text and bounded selection | optimized serial; exact substring scoring retained. |
| 7 | `relationships_for_entities` | Full edge scan per local query, `O(R)` | CPU | adjacency index, `O(degree(seed))` | optimized serial. |
| 8 | `ConnectedComponentsCommunityDetector.detect` | Build adjacency then BFS, `O(V + R)` plus deterministic sorting | CPU | already asymptotically appropriate; do not offload unless graph is very large | CPU / optional graph library; GPU is not justified for ordinary Python objects. |
| 9 | `SentenceTransformerEmbedder.embed_many` and `_embed_documents` | Original one document encode per call and repeated model calls, `O(N)` launch overhead | GPU/CPU numerical | native batched encode with device selection and synchronized CUDA-event timing | CUDA when present; CPU fallback. GPU compute is measured separately; internal framework transfer time is reported as unavailable rather than guessed. |
| 10 | `GlobalQueryEngine.query` | Independent map LLM calls then one reduce; `O(communities * LLM latency)` | I/O / remote inference | architecture is suitable for bounded async/distributed map tasks | asyncio/Ray/Dask; reduce remains sequential. |
| 11 | `persistence.save_graph` / chunks / checkpoints | JSON serialization and atomic writes grow with graph, `O(V + R)` per checkpoint | I/O | checkpoint frequency remains configurable; batch validation avoids extra unrelated work | Do not parallel-write one graph; coordinator owns persistence. |
| 12 | `DatasetDiscoverer.discover` | Reads a sample of every text file and CSV header, `O(files)` I/O | I/O | one-time discovery retained for correctness | modest document-level concurrency is possible, but small-file scheduling can dominate. |

## Measured baseline availability

The serial `baseline-mock` fixture completed with an actual wall-clock value in
its `summary.json`; its stage totals are deliberately also retained separately,
because stage totals are not valid end-to-end time under overlap. The fixture
has one tiny document and does not establish a production speedup.

## Acceleration environment validation

On 25 November 2026, this project environment detected an NVIDIA GeForce RTX
3050 Laptop GPU (4 GiB) through CUDA 12.8, and a CUDA tensor operation executed
on `cuda:0`. Ray 2.52.1 and Dask 2026.11.0 also completed the isolated
chunk-extraction smoke workload. The first real SentenceTransformer embedding
run could not complete because the model was not locally cached and the model
download timed out in this environment. No CUDA embedding speedup is reported
until the model is present and a same-workload CPU/CUDA benchmark has completed.

## Correctness design

Workers never mutate `KnowledgeGraph`. They return parsed entities, edges,
rejections, and usage data. The coordinator sorts by document/chunk position,
performs deduplication and provenance merge once, validates, checkpoints, and
persists. This avoids shared `last_rejections` and LLM usage state races.
