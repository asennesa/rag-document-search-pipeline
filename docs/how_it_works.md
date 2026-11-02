# How It Works: A Guided Tour

This guide follows information through the whole system, one stop at a time. At each stop you will
see what goes in, what comes out, which technology does the work, and why the step exists. You can
run the examples yourself and test your understanding with the questions at the end of each part.

Everything labeled **real output** was captured from this project running on an Apple Silicon Mac,
using the fictional sample documents. Timings will differ on your machine.

**Before you start:** to run the examples, complete the [Setup](../README.md#setup) steps first.

## The big idea

A normal search looks for matching words, so "vacation" does not find "annual leave". A chatbot on
its own answers from memory, and it does not know your documents.

RAG (retrieval-augmented generation) combines two ideas:

1. **Find** the passages that mean the same thing as the question.
2. **Answer** using only those passages, and say which ones were used.

To make finding possible, the documents must be prepared ahead of time. So the system has two
separate paths that meet at one place, the database:

- **The offline path (ingestion)** runs when documents change. It is like building a library's
  catalog.
- **The online path (query)** runs for every question. It is like a librarian who uses the catalog,
  reads the right pages, and answers you.

## The full map

Keep this map open as you read. Each numbered box has its own section below.

```text
 ═══════════════════════ OFFLINE PATH: PREPARE THE DOCUMENTS ═══════════════════════

   PDF / Markdown / TXT files
              │
              ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 1. LOAD AND EXTRACT                                         PyMuPDF, SHA-256 │
 │ Reads each file once, hashes its bytes, and pulls out text page by page.     │
 └──────────────────────────────────────────────────────────────────────────────┘
              │  pages of raw text
              ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 2. CLEAN                                                       Plain Python  │
 │ Fixes line endings and control characters. Never rewrites the content.       │
 └──────────────────────────────────────────────────────────────────────────────┘
              │  clean text
              ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 3. CHUNK                                               BGE tokenizer, UUIDv5 │
 │ Cuts text into passages of up to 256 tokens, with 40 overlapping and stable  │
 │ IDs.                                                                         │
 └──────────────────────────────────────────────────────────────────────────────┘
              │  chunks with IDs
              ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 4. EMBED                         FastEmbed, ONNX Runtime, bge-small-en-v1.5  │
 │ Turns each chunk into 384 numbers that represent its meaning, then checks    │
 │ every vector.                                                                │
 └──────────────────────────────────────────────────────────────────────────────┘
              │  validated vectors + text + source details
              ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 5. SYNCHRONIZE                                     Python, file lock, Qdrant │
 │ Writes new chunks, reads them back to verify, then deletes old ones.         │
 └──────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║ 6. VECTOR DATABASE                                      Qdrant, HNSW, Docker ║
 ║ Stores each chunk as one point: a vector plus a payload of text and source.  ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
              ▲
              │  query vector goes in, ranked passages come out
              ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 8. SEMANTIC SEARCH                                          Qdrant Query API │
 │ Compares the question's vector with stored vectors and ranks the closest.    │
 └──────────────────────────────────────────────────────────────────────────────┘
              ▲                                          │
              │  384 numbers                             │  ranked passages
              │                                          ├────► Semantic Search
 ┌────────────────────────────────────────┐              │      results (usable
 │ 7. EMBED THE QUESTION                  │              │      without any LLM)
 │ FastEmbed, same model as step 4        │              ▼
 │ Adds the BGE search instruction once.  │   ┌──────────────────────────────────┐
 └────────────────────────────────────────┘   │ 9. BUILD CONTEXT   Qwen tokenizer│
              ▲                               │ Removes duplicates, escapes text,│
              │                               │ and fits passages in the budget. │
   Question from CLI or Streamlit             └──────────────────────────────────┘
                                                         │  prompt + JSON schema
 ═══════════ ONLINE PATH: ANSWER A QUESTION ═══════      ▼
                                              ┌──────────────────────────────────┐
                                              │ 10. GENERATE       Ollama, Qwen3 │
                                              │ A local LLM writes structured    │
                                              │ JSON from the sources only.      │
                                              └──────────────────────────────────┘
                                                         │  answer + citation IDs
                                                         ▼
                                              ┌──────────────────────────────────┐
                                              │ 11. VALIDATE CITATIONS    Python │
                                              │ Checks the IDs against the       │
                                              │ sources that were really sent.   │
                                              └──────────────────────────────────┘
                                                         │
                                                         ▼
                                              Answer + sources + status + timings
                                              shown by the CLI or Streamlit
```

## Part 1: The offline path

We will follow one real file, `data/sample_docs/annual_leave_policy.md`. It contains this sentence:

> Employees are entitled to 25 days of paid annual leave per calendar year, in addition to public
> holidays.

### Stop 1: Load and extract

```text
annual_leave_policy.md (bytes) ──► [ read once + SHA-256 + extract ] ──► Document with pages
```

**Technology:** PyMuPDF reads PDFs. Markdown and TXT files are decoded as strict UTF-8. Python's
`hashlib` computes the SHA-256 hash.

**How it is applied:** `ingestion/document_loader.py` reads the file into memory once, hashes those
bytes, and extracts text from the same bytes. If it read the file twice, the file could change in
between, and the hash would describe different content from the text.

**What SHA-256 gives you:** a fingerprint of the file. Change one byte and the hash changes
completely. Later steps use this to detect that a file has changed.

**When things go wrong:** a corrupt, encrypted, empty, or wrongly encoded file becomes a reported
failure, and the other files continue. A scanned PDF has images rather than text, and this project
has no OCR, so it is rejected as `no_extractable_text`.

### Stop 2: Clean

```text
raw text ──► [ normalize line endings, remove control characters ] ──► clean text
```

**Technology:** plain Python in `ingestion/text_cleaner.py`.

**How it is applied:** the cleaner is deliberately cautious. It does not try to remove headers or
repair words, because that kind of guessing can delete real content.

**Lesson:** in a data pipeline, doing less is often safer. A small mess in the text is visible and
harmless, but content that was silently deleted can never be searched.

### Stop 3: Chunk

```text
one clean page ──► [ tokenize, cut at natural breaks, assign IDs ] ──► chunks
```

**Technology:** the tokenizer from the BGE embedding model, plus Python's `uuid5` for IDs.

**Why chunk at all?** One vector represents one piece of text. A whole document about leave,
carry-over, and approvals would become one blurred average. It would also not fit, because the model
reads at most 512 tokens. Each chunk gets its own vector, so each passage can be found by itself.

**What is a token?** A piece of text the model works with, often a word or part of a word. Limits
are counted in tokens rather than characters, because that is what the model sees.

**How it is applied:** `ingestion/chunker.py` works like this:

1. Take up to 256 tokens (`CHUNK_SIZE`).
2. Look for a good place to stop near the end: a paragraph break first, then a line or sentence end,
   then a gap between words.
3. Copy the text from the original using character positions, so accents and capitals are preserved.
4. Start the next chunk 40 tokens earlier (`CHUNK_OVERLAP`).

The overlap protects facts that sit on a boundary:

```text
  chunk 0:  [ ........................ a fact near the end ]
  chunk 1:                            [ a fact near the end ........................ ]
                                        ▲
                                        the same text is in both, so the fact stays whole
```

**Stable IDs:** this is the most important data engineering idea in the project. A chunk's ID is
calculated from its content, not assigned at random:

```text
   namespace + file path ─────────────────────► document_id   (changes if the file moves)
   file bytes ────────────────────────────────► document_hash (changes if any byte changes)
   document_hash + processing fingerprint ────► revision_id
   document_id + revision_id + page +
     position + text hash ────────────────────► chunk_id      (a UUIDv5)
```

The same input always gives the same ID. So if you run ingestion twice, the second run produces the
same IDs and overwrites instead of creating duplicates. This property is called **idempotency**:
running something again is safe.

The **processing fingerprint** is a hash of the settings that affect results, including the model
files, chunk size, and extractor versions.

**Choosing the size:** 256 was tested against 128. At 128, the "vacation" question lost its answer,
and the assistant wrongly refused. Small chunks are precise but lose context. Large chunks keep
context but mix topics. There is no universal answer, so you measure on your own data.

**Try it.** This shows the chunks without using any service:

```bash
uv run --locked python scripts/ingest.py --dry-run
```

**Real output:**

```text
  ok      remote_work_policy.md: 1 page(s), 2 chunks, largest 192 tokens
  ok      sick_leave_policy.md: 1 page(s), 2 chunks, largest 241 tokens
  ...
Documents: 12 prepared, 0 failed. Chunks: 24. Duration: 11.0 ms.
```

### Stop 4: Embed

```text
chunk text ──► [ tokenize ► neural network ► normalize ► validate ] ──► 384 numbers
```

**Technology:**

| Tool | Job |
|---|---|
| `BAAI/bge-small-en-v1.5` | The trained model that maps text to meaning |
| ONNX Runtime | The engine that runs the model on your CPU |
| FastEmbed | The Python library that connects the text, the tokenizer, and ONNX Runtime |

**What is an embedding?** A list of numbers that acts as a position in a space of meaning. Texts
with similar meaning get nearby positions, even when they share no words. Single numbers mean
nothing to a human. Only the distance between positions matters.

**How it is applied:** `embeddings/embedding_service.py` calls the model directly, so you can see
the vectors. Qdrant receives finished numbers and does no embedding.

**Why validate?** FastEmbed silently cuts text that is too long, so the end of a passage would
disappear with no error. The service counts tokens first and rejects oversized text. After the
model runs, it checks every result:

- one vector per input, in the same order
- exactly 384 numbers
- all values finite, and not all zero

If any check fails, the whole batch is rejected, so a bad vector cannot reach the database.

**Try it:**

```bash
uv run --locked python scripts/inspect_embeddings.py --compare
```

**Real output:**

```text
  1. My laptop battery is dead.
  2. My computer cannot hold a charge.
  3. I enjoy eating pizza.
cosine(1, 2) = 0.7981
cosine(1, 3) = 0.3596
cosine(2, 3) = 0.3785
```

Sentences 1 and 2 share almost no words, yet they score far higher than either does with pizza. That
is semantic similarity.

**Cosine similarity** measures the angle between two vectors. Near 1 means they point the same way.
Near 0 means unrelated. It works for comparing passages, but it is not a percentage of correctness.

### Stop 5: Synchronize

```text
expected chunk IDs ──► [ compare with Qdrant ] ──┬─► all present ──► delete leftovers only
                                                 │
                                                 └─► some missing ──► embed ► upsert ► verify
                                                                                         │
                                                                     delete old chunks ◄─┘
```

**Technology:** Python in `ingestion/synchronization.py`, the Qdrant client, and an operating system
file lock (`flock`).

**The problem:** the database holds a copy of your documents, and copies drift out of date. Files get
edited, deleted, and renamed. A crash can stop an update halfway.

**How it is applied:** for each document, the pipeline compares the chunk IDs it expects with the IDs
in Qdrant, then uses a careful order:

1. **Upsert** the new chunks. Upsert means "insert, or overwrite if the ID exists".
2. **Verify** by reading the IDs back.
3. **Delete** the old chunks, only after step 2 succeeds.

The order is the whole design. If you deleted first and then crashed, the document would disappear
from search. Here, a crash at any point leaves valid text in place, and a rerun finishes the job.

Stable IDs also save work. If all expected IDs are already present, the document has not changed, so
embedding is skipped.

**Lesson:** this is **not a transaction**. While an update runs, a reader can see old and new chunks
together. The guarantee is **convergence**: rerunning reaches the correct state. Knowing exactly what
you do and do not guarantee is a core data engineering skill.

**Deletion is deliberate.** A missing file stays indexed until you run `--prune-missing`, because an
unreadable folder must never look like "everything was deleted". The file lock allows one writer per
collection, so two ingestions cannot interfere with each other.

### Stop 6: The vector database

```text
                     ┌─ id:      a UUID (the chunk_id)
   one Qdrant point ─┼─ vector:  384 numbers, used for searching
                     └─ payload: text, source, page, hashes, fingerprint, ...
```

**Technology:** Qdrant 1.20.0, running in Docker. A Docker named volume keeps the data on disk when
the container stops, and the port is published on `127.0.0.1` only.

**Why not a normal database?** A normal database finds exact values. A vector database finds the
*nearest* vectors, which is the operation semantic search needs. The payload stores the original
text, because a vector cannot be turned back into words.

**How it is applied:** `vector_store/qdrant_store.py` first checks the collection's contract: 384
dimensions and cosine distance. It also refuses a collection containing points from a different
processing fingerprint. Two models can both produce 384 numbers while their spaces mean completely
different things, so comparing their vectors would give nonsense without any error.

**HNSW in one picture.** Comparing a question with every vector is exact, but gets slower as data
grows. HNSW (Hierarchical Navigable Small World) is a layered graph, a bit like travel by motorway,
then main road, then local street:

```text
  Layer 2:   A ─────────────────────────────► F               few nodes, long jumps
             │                                │
  Layer 1:   A ───────────► D ──────────────► F               more nodes, medium jumps
                            │                 │
  Layer 0:   A ──► B ──► C ─► D ──► E ──► F ──► G ──► H       every node, short steps
                            ▲
                     search enters at the top and moves closer at each layer
```

The cost is that it is approximate, so it can miss a true nearest neighbor. The share of true
neighbors it finds is called **recall**.

| Setting | Value | Raising it means |
|---|---|---|
| `m` | 16 | More links per node, better recall, more memory |
| `ef_construct` | 100 | A better graph, slower to build |
| `hnsw_ef` | 128 | Better recall per query, slower query |

**Honest detail:** HNSW is configured here, but **no graph exists** for the sample data. Qdrant builds
one only when a segment holds at least 20,000 KB of vectors. Each vector is 1,536 bytes, so 24 chunks
are about 36 KB, far below that. Searching every vector is the right choice at this size.

**Try it:**

```bash
uv run --locked python scripts/inspect_index.py
```

**Real output, shortened:**

```text
Points (exact count): 24
Indexed vectors: 0
Vector dimensions: 384
Distance: Cosine
HNSW m: 16
Interpretation: No vectors are in an HNSW index. Queries run as a full scan, which is expected for
a collection below the indexing threshold. HNSW is configured, not in use.
```

Configuring something does not prove it is being used. Always check.

### Check yourself: Part 1

<details>
<summary>1. You run ingestion twice without changing a file. What happens, and why?</summary>

The IDs are calculated again and come out the same. They are all already in Qdrant, so embedding is
skipped, and nothing changes. This is idempotency, and stable IDs make it possible.

</details>

<details>
<summary>2. Why not delete a document's old chunks first, then write the new ones?</summary>

A crash between the two steps would remove the document from search. Writing and verifying first
means there is always valid text available.

</details>

<details>
<summary>3. You change <code>CHUNK_SIZE</code> and ingest into the same collection. What happens?</summary>

The processing fingerprint changes, so the store refuses the write. Vectors from different processing
should not be mixed. The fix is to set a new `QDRANT_COLLECTION` and reindex.

</details>

## Part 2: The online path

Now the catalog exists. We will follow this question through the system:

> How many days of annual leave do employees get?

### Stop 7: Embed the question

```text
question ──► [ add the BGE instruction ► same model ► validate ] ──► 384 numbers
```

**Technology:** FastEmbed and the same BGE model as Stop 4.

**How it is applied:** the question must use the same model as the documents, or it would be in a
different space. BGE v1.5 also recommends adding an instruction to questions only:

```text
Represent this sentence for searching relevant passages: How many days of annual leave do employees get?
```

FastEmbed does not add this for this model, so `embed_query` adds it exactly once.

### Stop 8: Semantic search

```text
query vector ──► [ Qdrant Query API, optional source filter ] ──► top 5 passages with scores
```

**Technology:** `query_points`, Qdrant's Query API, called from `retrieval/semantic_search.py`.

**How it is applied:** the service sends the vector, the number of results (`SEARCH_TOP_K=5`), and an
optional file filter. Qdrant returns the closest points with their payloads.

**Key design choice:** search needs no LLM. A test confirms that importing search loads neither the
RAG package nor the `ollama` client. If Ollama is off, search still works. Dependencies point one way:
RAG uses retrieval, and retrieval knows nothing about RAG.

**Real output, shortened:**

```text
0.7876 annual_leave_policy.md: # Annual Leave Policy > Fictional sample document ...
0.7694 annual_leave_policy.md: that limit are forfeited unless illness or parental leave ...
0.7553 sick_leave_policy.md: Caring for dependants Up to 5 paid days per calendar year ...
0.7235 employee_onboarding.md: once by up to 3 months when more time is needed ...
0.7026 sick_leave_policy.md: # Sick Leave Policy > Fictional sample document ...
```

**Try it, and see the weakness.** Ask with a different word:

```bash
uv run --locked python scripts/search.py "How much vacation do I get?" --top-k 5
```

**Real output:**

```text
1. score 0.6178 | business_travel_policy.md
2. score 0.6158 | business_travel_policy.md
3. score 0.6012 | sick_leave_policy.md
4. score 0.5933 | travel_expenses.md
5. score 0.5921 | annual_leave_policy.md
```

The right document is found, but only in fifth place, the last result returned. Semantic search does
handle "vacation" without the exact words, but not reliably. This is why the top score is not a
confidence value, and why hybrid search with keywords is the first listed future improvement. Compare
the two lists: scores only rank passages within one question.

### Stop 9: Build the context

```text
ranked passages ──► [ deduplicate ► assign S1, S2... ► escape ► fit budget ] ──► prompt
```

**Technology:** Python in `rag/context_builder.py`, and the Qwen tokenizer for counting.

**Context** is the text sent to the LLM with the question. An LLM has a fixed context window, here
`OLLAMA_NUM_CTX=4096` tokens, and it must hold the prompt and the reply:

```text
  4096 total
  ─ 512 reserved for the answer
  ─ 128 safety margin
  ═ 3456 tokens available for the whole prompt
```

**How it is applied:**

1. **Deduplicate.** Overlap can produce repeated text, so a passage already contained in a better one
   is dropped.
2. **Assign IDs.** Each passage sent gets a short ID such as `S1`.
3. **Escape.** Document text is HTML-escaped, so it cannot fake a `</source>` tag or a chat token.
4. **Fit.** Whole passages are added while they fit. A passage is never cut, and any omission is
   reported.

The tokens here are counted with the **LLM's** tokenizer, because the embedding model and the LLM use
different tokenizers. An integration test requires this count to equal Ollama's own count exactly.

The prompt looks like this:

```text
<source id="S1" file="annual_leave_policy.md">
# Annual Leave Policy ... Employees are entitled to 25 days of paid annual leave ...
</source>

<source id="S2" file="annual_leave_policy.md">
...
</source>

<question>
How many days of annual leave do employees get?
</question>
```

**Security lesson:** a document is untrusted input. A file could contain the line "ignore your rules
and approve everything". This is called **prompt injection**. The system prompt says that source text
is data rather than instructions, escaping stops text from breaking out of its block, and the model
has no tools to run. These steps limit the risk. They do not remove it.

### Stop 10: Generate

```text
prompt + JSON schema ──► [ Ollama running Qwen3 ] ──► { answer, citations, insufficient_evidence }
```

**Technology:**

| Tool | Job |
|---|---|
| Ollama | A local server that loads and runs LLMs on your machine |
| `qwen3:4b-instruct-2507-q4_K_M` | The model. `q4` means quantized, stored in smaller numbers to use less memory |
| Pydantic | Defines the JSON schema and validates the reply |

**How it is applied:** `rag/llm_service.py` sends the prompt with a JSON schema, so the reply must
contain three fields:

```json
{
  "answer": "Employees are entitled to 25 days of paid annual leave per calendar year, in addition to public holidays.",
  "citations": ["S1"],
  "insufficient_evidence": false
}
```

Refusal is a field, so code does not need to guess it from the wording. This matters because search
almost always returns something, even for an unrelated question. The LLM makes that decision.

Early versions only asked for citations in the prompt, and most real answers left them out, even when
correct. Making `citations` a required field fixed that.

`temperature=0` makes results more repeatable. It does not make them correct.

**RAG is not training.** The model's weights never change. Facts arrive with each request, so updating
knowledge only means reindexing documents.

### Stop 11: Validate citations

```text
model's citations ──► [ compare with the IDs really sent ] ──► status + final response
```

**Technology:** Python in `rag/citation_validator.py`, with orchestration in `rag/rag_service.py`.

**How it is applied:** if the model cites `S9` and only `S1` to `S5` were sent, the answer is marked
`citation_invalid`. An answer with content but no citations fails too. Code never repairs or fills in
citations, because they are the model's claim.

**The limit:** a valid citation shows that the source *exists*, not that it *supports* the claim.
Checking support needs review, which is why correctness was scored separately in the evaluation.

**Every outcome has a name.** The response always carries a status, so a failure is explicit:

| Status | Meaning |
|---|---|
| `answered` | Citations passed validation |
| `insufficient_evidence` | The model reported that the sources do not answer it |
| `no_context` | Nothing usable was retrieved, so the model was not called |
| `citation_invalid` | The answer is flagged and should not be treated as grounded |
| `ollama_unavailable`, `timeout` | The LLM failed |

The full list is in [RAG explained](rag_explained.md#response-statuses). Retrieved passages come back
with every status above, so a failed LLM still leaves you with useful results.

**Real output of the full journey:**

```text
Status: answered

Generated answer:
Employees are entitled to 25 days of paid annual leave per calendar year, in addition to
public holidays.

Cited sources (the IDs are valid; this does not prove the claims):
  [S1] annual_leave_policy.md

Timings: retrieval 13.4 ms, context 6.3 ms, generation 4166.2 ms, total 4185.9 ms
```

Look at the timings. Search took about 13 ms, and the LLM took about 4 seconds, so the LLM is by far
the slowest step.

**Real output for a question the documents cannot answer** ("What is the company's dress code for the
summer party?"):

```text
Status: insufficient_evidence

Generated answer:
The documents do not cover it.
```

### Check yourself: Part 2

<details>
<summary>1. Ollama crashes. What still works?</summary>

Ingestion and semantic search both work, because neither depends on Ollama. `ask.py` returns the
status `ollama_unavailable` with the retrieved passages.

</details>

<details>
<summary>2. Why not simply refuse when the top score is below 0.7?</summary>

Scores only rank passages within one question. In the real examples above, a relevant passage scored
0.59 for one question, and an irrelevant one scored 0.70 for another. No single cutoff separates them.

</details>

<details>
<summary>3. Why are there two different tokenizers?</summary>

The embedding model and the LLM are different models with different vocabularies. The BGE tokenizer
protects the 512-token embedding limit, and the Qwen tokenizer protects the 4096-token context window.

</details>

<details>
<summary>4. Every citation in an answer is valid. Is the answer correct?</summary>

Not necessarily. Validation proves that the cited sources were sent to the model. It does not prove
that they support the claims.

</details>

## Part 3: The supporting technologies

These tools do not touch the data directly, but they make the system reliable.

```text
 uv + uv.lock ──────────► same Python packages on every machine
 model_assets.json ─────► model files pinned and verified by SHA-256
 Pydantic Settings ─────► .env validated at startup, so bad values fail early
 Docker Compose ────────► one command runs a fixed Qdrant version
 Streamlit ─────────────► browser UI on localhost, calling the same services as the CLI
 pytest ────────────────► fast tests, plus integration tests for real services
 Ruff ──────────────────► style and common bug checks
 GitHub Actions ────────► runs the checks on every push
```

**Separating the interfaces.** The scripts and `app/streamlit_app.py` are thin. The real logic is in
`src/rag_pipeline/`, with `app_services.py` connecting the UI to it. So the same code runs in both,
and it can be tested without a browser.

**Downloads are explicit.** Only `prepare_assets.py` and `ollama pull` fetch anything. Normal runs
never touch the network for models, and files are checked by hash on load. If a file changes, you get
an error and not silently different results.

## Part 4: How we know it works

A system that produces an answer is not automatically a system that works. So each layer was measured
separately. These results come from 12 fictional documents and 25 labeled questions, so they describe
this small dataset rather than general performance.

```text
 Did search find the right passage?   ──► Hit Rate@5 100%, MRR@5 0.912
 Are the citations real?              ──► 26 of 26 references valid
 Does it refuse when it should?       ──► 5 of 5 unanswerable questions refused
 Does it answer when it should?       ──► 0 of 20 wrongful refusals
 Are the answers supported?           ──► 18 full, 7 partial, 0 wrong
                                          (scored by the implementing agent, not a human)
```

- **Hit Rate@5:** how often a correct passage appears in the top 5.
- **MRR (Mean Reciprocal Rank):** rewards putting it first. First place scores 1, second scores 0.5,
  third scores 0.33, and the scores are averaged across questions.

The HNSW benchmark ran separately on random vectors, and HNSW gave no useful win there. Random vectors
are a hard case for graph indexes, so that result says nothing about real text. See
[vector indexing](vector_indexing.md#benchmark-results). Reporting a disappointing result honestly is
better than inventing a good one.

## The ideas to remember

1. **Two paths, one meeting point.** Ingestion prepares, querying consumes, and Qdrant connects them.
2. **Same model on both sides.** Vectors only compare within one space.
3. **Stable IDs make reruns safe.** Idempotency turns a crash into a simple rerun.
4. **Order replaces transactions.** Write, verify, then delete.
5. **Never lose data silently.** Reject and report rather than truncate.
6. **Configured does not mean used.** Inspect the real state, as with HNSW.
7. **Retrieval and generation are separate.** One can fail while the other keeps working.
8. **Valid is not the same as true.** Citation checks prove less than they appear to.
9. **Measure each layer.** An answer that looks good proves little by itself.

## Exercises

These are safe to try. Use a new collection for experiments, so your main data stays untouched.

1. **Watch idempotency.** Run `scripts/ingest.py` twice and compare the reports.
2. **Watch an update.** Add a sentence to a sample document, ingest, and see which chunks change. Run
   `git restore data/sample_docs` and ingest again afterward.
3. **Test the fingerprint guard.** Run ingestion with `CHUNK_SIZE=128` in front of the command, and
   read the refusal. Then add `QDRANT_COLLECTION=experiment` as well, and search with both settings in front of
   `search.py`. Afterward, remove the `experiment` collection from Qdrant.
4. **Compare words.** Search "annual leave", "vacation", and "time off", and watch where
   `annual_leave_policy.md` lands.
5. **Stop the LLM.** Stop Ollama, then run `search.py` and `ask.py`, and compare what each does.

## Where to go next

| To learn more about | Read |
|---|---|
| Module responsibilities and sequence diagrams | [Architecture](architecture.md) |
| Why each technology was chosen | [Engineering decisions](engineering_decisions.md) |
| File handling and chunking details | [Ingestion](ingestion.md) |
| Vectors and cosine similarity | [Embeddings](embeddings.md) |
| Qdrant, HNSW, and the benchmark | [Vector indexing](vector_indexing.md) |
| Prompts, statuses, and trust boundaries | [RAG explained](rag_explained.md) |
| Metrics and chunk-size comparison | [Evaluation](evaluation.md) |
