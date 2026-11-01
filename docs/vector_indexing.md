# Vector storage and indexing

Qdrant stores each chunk as one point: the 384-dimensional vector, plus a payload with its
text and source references. Vectors arrive already computed by `EmbeddingService`; Qdrant's
automatic embedding features are not used.

## Exact and approximate search

**Exact search** compares the query with every vector. It always finds the true nearest
neighbors, and its cost grows with the collection size.

**Approximate nearest-neighbor (ANN) search** visits only a small part of the data using an
index. It is much faster at scale, but it can miss a true neighbor. The share of true
neighbors it finds is called recall.

## HNSW

HNSW stands for Hierarchical Navigable Small World. It is a layered graph in which each
vector links to some of its neighbors.

- **Upper layers** are sparse, with long links that cross the space quickly.
- **Lower layers** are denser, and the bottom layer holds every indexed vector.
- **A search** enters at the top, moves greedily toward vectors closer to the query, and
  drops a layer whenever it can get no closer, refining as it descends.

| Setting | Where | Effect of raising it |
|---|---|---|
| `m` | Collection | More links per node: better recall, more memory, slower build |
| `ef_construct` | Collection | Wider search during build: better graph, slower build |
| `hnsw_ef` | Each query | Wider search per query: better recall, slower query |
| `full_scan_threshold` | Collection | Query planner prefers a scan on more data |
| `indexing_threshold` | Optimizer | Segments must be larger before a graph is built |

Both thresholds are measured in **kilobytes of vector data, not points**. One 384-dimension
float32 vector takes 1,536 bytes, so the default 20,000 KB threshold means about 13,000
vectors per segment before graph data and payload overhead.

These defaults (`m=16`, `ef_construct=100`, `hnsw_ef=128`) are starting values, not tuned
results. Measurements belong to the separate benchmark in Phase 8.

## What this project's collection actually does

```bash
uv run --locked python scripts/inspect_index.py
```

The sample collection is far below the indexing threshold, so **HNSW is configured but no
graph is built**. Inspection reports zero indexed vectors and queries run as a full scan,
which is the correct behavior at this size.

Evidence has limits:

- `indexed_vectors` above zero shows that an index exists, not that a given query used it.
- Qdrant plans each query per segment, and filters can change the plan.
- `exact=False` permits approximate search; it does not force graph traversal.
- A `green` status does not mean indexing happened.

`exact=True` always requests a full comparison, which gives the benchmark its ground truth.

## Payloads and collection contract

Keyword indexes exist on `source_namespace`, `document_id`, `source`, `revision_id`, and
`processing_fingerprint`, so filters stay fast. Payload fields are validated on write and
on read.

Before writing, the store checks for one unnamed cosine vector of 384 dimensions. It also
counts points from any other processing fingerprint, including points without one. Equal
dimensions do not make vectors comparable, so a mismatch is refused without changes. The
fix is to set a new `QDRANT_COLLECTION` and reindex. Collections are never recreated
automatically.

## Keeping Qdrant in sync with files

```bash
uv run --locked python scripts/ingest.py
```

For each document the pipeline reads the indexed IDs and compares them with the complete
expected set. Matching IDs matter, not matching counts: IDs derive from content.

1. All expected chunks present: skip embedding entirely.
2. Otherwise embed everything, upsert in acknowledged batches, and re-read the IDs.
3. Only once every expected ID is confirmed, delete the document's obsolete IDs.

| Interrupted during | State left behind | Next run |
|---|---|---|
| Embedding | Unchanged | Starts again |
| Upsert | Old chunks plus some new ones | Upserts the same IDs again |
| Verification | Old and new chunks | Upserts or cleans up |
| Delete | New chunks plus some old ones | Skips embedding, finishes deleting |

**This is not a transaction.** Qdrant does not make these steps atomic, so during an update
a reader can see old and new chunks together. The guarantee is convergence: rerunning
reaches the correct state and valid text is never removed first.

- **Invalid update:** the previous revision stays searchable and is reported as stale.
- **Deleted file:** it stays indexed and is reported as missing until explicitly pruned.

```bash
uv run --locked python scripts/ingest.py --prune-missing --dry-run
uv run --locked python scripts/ingest.py --prune-missing
uv run --locked python scripts/ingest.py --delete-source annual_leave_policy.md
```

Pruning requires a complete listing of the configured directory. An unreadable folder, an
empty listing, or `--documents-dir` refuses it, because each could make present documents
look deleted. A file that exists but fails to parse is not treated as missing.

A file lock allows one writer per collection on one host. The operating system releases it
if the process dies. It does not coordinate different machines.

## Known limitations

- One writer per host; multi-host writers are out of scope.
- An edited document is fully re-embedded, even if only one chunk changed.
- Readers can briefly see mixed revisions during an update.
- A renamed file is a new document plus a missing one until pruned.
- Inspection cannot show which path an individual query took.
