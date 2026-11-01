# Interview walkthrough

A five-to-seven-minute demo. Before starting, run Qdrant and Ollama, ingest the samples, and open the
app and a terminal. Timings on screen vary by machine, so read them out rather than quoting numbers.

## 1. The problem (30 seconds)

“This is a local RAG pipeline. RAG looks like an LLM problem, but most of what goes wrong is data
engineering: stale indexes, lost text, and claims nobody verified. So I built each stage by hand,
with no framework.”

## 2. Ingestion and idempotency (75 seconds)

```bash
uv run --locked python scripts/ingest.py
```

Run it again and point out `12 unchanged` and `Vectors upserted: 0`.

“Chunk IDs come from content, so reruns can't duplicate. An edit writes and verifies new chunks before
deleting old ones. If it fails midway, old text survives and a rerun finishes. It isn't a transaction,
and readers can briefly see both versions, but it always converges.”

Chunking question to raise: FastEmbed silently truncates, so I count with a separate untruncated
tokenizer, which is why no text can disappear unnoticed.

## 3. Where text becomes vectors (45 seconds)

```bash
uv run --locked python scripts/inspect_embeddings.py --compare
```

“The battery sentences score far closer to each other than to the pizza sentence, and they share
almost no words. I also fingerprint the model and settings, because two models with 384 dimensions
are still different vector spaces. Search refuses a mismatched collection, since it would otherwise
return confident nonsense.”

## 4. HNSW, honestly (75 seconds)

```bash
uv run --locked python scripts/inspect_index.py
```

Point out `Indexed vectors: 0`.

“HNSW is configured here, but no graph exists. With 24 chunks, Qdrant correctly does a full scan.
Configuration isn't evidence, so the tool reports what's observed.”

Then show the benchmark table in `docs/vector_indexing.md`.

“On random vectors, HNSW didn't really win: at 50,000 points it was faster only at very low recall.
I kept that result. Random vectors are a known worst case, so it doesn't predict real text, but
exact search being fine at this scale is a useful engineering conclusion in itself.”

## 5. Search and RAG in the app (90 seconds)

In **Semantic Search**, ask “Can I work remotely?” and show scores and sources. Then in **RAG
Assistant**, ask “How much can I spend on a hotel and on meals during a business trip?”

“The answer cites two documents, and evidence sits apart from generated text. Citation IDs are
checked in code against what was actually sent to the model.”

Then ask “What is the policy on underwater vacations?” and show the refusal.

If you want a strong moment, stop Ollama and show that search still works.

## 6. Evaluation (60 seconds)

Open `docs/evaluation.md`.

“Two findings I like. First, an early test suggested smaller chunks were better. On the labeled set
they ranked better but lost the ‘vacation’ question, causing a wrongful refusal, so I kept the
default. Second, one answer had 100% valid citations and still didn't cite the document its claim
came from. Valid citations aren't correctness, which is why correctness needs a reviewer.”

## 7. Close (30 seconds)

“The next step would be hybrid search for vocabulary mismatches. At larger scale this maps onto Delta
Lake and Databricks AI Search, but I'd want a real need first.”

## Likely questions

| Question | Short answer |
|---|---|
| Why no LangChain? | To show exactly where each step happens, and to control failure behavior |
| Why Qdrant? | Real HNSW controls, payload filters, and a simple local deployment |
| How do you stop hallucinations? | You can't fully. Grounding, refusal, and citation checks reduce them, and evaluation measures them |
| What about prompt injection? | Escaping stops structural breakout and the model has no tools, but plain text can still persuade it |
| What breaks first at scale? | One writer per host and full re-embedding of edited documents |
| What are you least sure of? | The evaluation is small, I wrote both questions and documents, and correctness was scored by an AI agent, not a person |
