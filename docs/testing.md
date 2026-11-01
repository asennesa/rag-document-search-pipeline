# Testing

| Suite | Command | Needs | In CI |
|---|---|---|---|
| Unit | `uv run --locked pytest` | Nothing; network blocked | Yes |
| Qdrant store, benchmark protocol | `uv run --locked pytest tests/integration/test_qdrant_store.py tests/integration/test_benchmark_qdrant.py` | Qdrant | Yes |
| Tokenizers | `uv run --locked pytest tests/integration/test_embedding_tokenizer.py tests/integration/test_generation_tokenizer.py` | Prepared assets; Ollama for count checks | No |
| Embeddings | `uv run --locked pytest tests/integration/test_embeddings.py` | Embedding model | No |
| Ingestion, search, evaluation | `uv run --locked pytest tests/integration/test_ingestion_sync.py tests/integration/test_semantic_search.py tests/integration/test_evaluation.py` | Qdrant and model | No |
| RAG | `uv run --locked pytest tests/integration/test_rag_ollama.py` | Also Ollama and its model | No |
| End to end | `uv run --locked pytest tests/integration/test_end_to_end.py` | Everything | No |

Run everything with `uv run --locked pytest tests/unit tests/integration`, and add `-rs` to see skips.

**A skip is not a pass.** Without Ollama, its tests skip with `BLOCKED, not passed`.

Integration tests use disposable collections and temporary files, never the application collection.
They do not start services or download assets. CI skips model tests because downloading models on
every run is slow; the Ollama model is never downloaded there.

## What a passing test does and does not show

- Unit tests prove application logic. Mocked tests say nothing about real model behavior.
- Local-mode Qdrant checks adapter behavior only. HNSW, optimizer, and persistence evidence comes
  from the real server.
- The live browser checks in `PLANS.md` were run with a throwaway tool and are not part of the suite.
- Prompt token counting is checked for exact equality with Ollama, so a changed template is caught.

During development, safeguards were deliberately broken to confirm tests fail when they should.
That found real gaps, such as nothing proving that search covered uploads.
