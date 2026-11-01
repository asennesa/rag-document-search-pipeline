# Embeddings

An embedding turns text into a fixed-length list of numbers, so that texts with similar
meaning end up close together. This project makes that step explicit in
`src/rag_pipeline/embeddings/embedding_service.py`; Qdrant receives finished vectors.

```python
chunks = chunker.split_document(document, revision_id)
texts = [chunk.text for chunk in chunks]
vectors = embedding_service.embed_documents(texts)
```

## From text to a vector

The model is `BAAI/bge-small-en-v1.5`, run locally through FastEmbed and ONNX Runtime.

1. The tokenizer splits text into tokens and adds `[CLS]` and `[SEP]`.
2. The BERT encoder gives each token a 384-dimensional representation in context.
3. FastEmbed keeps the `[CLS]` output as the vector for the whole text.
4. FastEmbed scales the vector to length 1.

No single dimension has a readable meaning. Meaning lies in position relative to other
vectors, so vectors are comparable only if they come from the same model and processing.
A different model with 384 dimensions is still a different space, which is why the
processing fingerprint records artifact hashes as well as the dimension count.

## Cosine similarity

Cosine similarity measures the angle between two vectors:

```text
cosine(a, b) = (a · b) / (|a| × |b|)
```

Values near 1 point the same way, and values near 0 are unrelated. FastEmbed's vectors have
length 1, so the score reduces to the dot product. A score ranks passages for one query;
it is not a probability or a measure of whether a passage answers the question.

See measured values on your machine:

```bash
uv run --locked python scripts/inspect_embeddings.py --compare
```

## Documents and questions

BGE v1.5 recommends a retrieval instruction on the question side only. FastEmbed's
`query_embed` does not add it for this model, so `embed_query` adds it exactly once and
counts it toward the token limit. `passage_embed` receives chunk text unchanged.

## What `EmbeddingService` guarantees

| Check | Behavior on failure |
|---|---|
| Input is nonblank and within 512 tokens, counted without truncation | Rejected before inference |
| One vector per input, in order | Whole batch rejected |
| Exactly 384 real numeric values | Whole batch rejected |
| Finite after conversion to float32, nonzero | Whole batch rejected |

FastEmbed returns a generator, so inference runs while it is consumed. The service consumes
it fully before validating, which means a failure midway returns nothing rather than part
of a batch. Output is plain Python floats, ready for Qdrant. Errors carry positions, not
text, and logs carry only counts and durations.

## Assets and offline use

```bash
uv run --locked python scripts/prepare_assets.py --embedding-model
```

This is the only command that downloads the weights, about 64 MB on disk. The ONNX file
and tokenizer come from one commit of `Qdrant/bge-small-en-v1.5-onnx-Q`, pinned with SHA-256
hashes in the tracked `model_assets.json`. Each load verifies those hashes and passes
FastEmbed a local snapshot path, which skips its download logic. A missing or changed file
fails with this command as the fix. To upgrade deliberately, remove the manifest entries,
prepare again, and reindex into a new collection.

## Inspecting vectors

```bash
uv run --locked python scripts/inspect_embeddings.py "Employees receive 25 days of annual leave."
uv run --locked python scripts/inspect_embeddings.py --query "How much vacation do I get?"
```

Load time is reported apart from generation time, because the first is paid once per
process. Timings vary by machine and are not benchmarks.

## Known limitations

- Inference runs on the CPU provider only; batch size and threads are starting values.
- English is the validated language scope.
- Vectors in a batch can differ from single-text results by tiny amounts because of padding.
