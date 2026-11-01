# RAG Document Search Pipeline

A local Python portfolio project for explicit document embeddings, Qdrant vector
indexing, semantic retrieval, and Ollama generation with source citations.

**Current scope: Phases 1-6.** Documents flow from files through chunking and validated
embeddings into Qdrant. Semantic search retrieves ranked passages without any LLM, and the
RAG assistant answers through local Ollama with validated source IDs. Streamlit screens are
scheduled in a later approved phase. See
[PLANS.md](PLANS.md) for the architecture, acceptance criteria, and progress.

## Setup

Target: Python 3.12 on macOS Apple Silicon or Ubuntu 24.04. Docker Desktop/Engine
with the Compose plugin is required for Qdrant. Python runs on the host.

Install [uv 0.13.6](https://docs.astral.sh/uv/getting-started/installation/).
This workstation has it at `~/.local/bin/uv`; add that directory to the current shell:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12.16
uv sync --locked
cp .env.example .env
uv run --locked python -c 'from rag_pipeline.config import Settings; Settings(); print("Configuration valid")'
```

Run commands from the repository root. Skip the copy if `.env` already exists.
Dependency installation includes later-phase libraries but downloads no embedding
or LLM models. Imports and configuration construction do not initialize services.

Settings precedence is constructor arguments, environment variables, `.env`, then
defaults. Environment names are case-insensitive. Unknown `.env` keys are rejected
to catch typos. Paths are relative to the working directory and are not created by
configuration validation. Keep `.env` and private documents out of Git.

## Document preparation

```bash
uv run --locked python scripts/prepare_assets.py --embedding-tokenizer
uv run --locked python scripts/ingest.py --dry-run
```

The first command downloads only tokenizer files, with no model weights, into the
ignored `.cache/` directory and pins them in `model_assets.json`. The dry run extracts and
chunks the 12 fictional policies in `data/sample_docs/`. It needs no running service and
writes nothing. It accepts `--documents-dir` for another folder. PDF, Markdown, and TXT are
supported; scanned PDFs need OCR, which is not supported. See
[ingestion and chunking](docs/ingestion.md).

## Embeddings

```bash
uv run --locked python scripts/prepare_assets.py --embedding-model
uv run --locked python scripts/inspect_embeddings.py "Employees receive 25 days of annual leave."
uv run --locked python scripts/inspect_embeddings.py --compare
```

The first command downloads the ONNX weights, about 64 MB, into `.cache/` and pins their
hash. Inspection then runs locally with no network, Qdrant, or Ollama. Dry-run ingestion
does not need the weights. See [embeddings](docs/embeddings.md).

## Qdrant

Start Docker, then run:

```bash
docker compose config --quiet
docker compose up -d qdrant
curl -fsS --retry 15 --retry-all-errors --retry-delay 1 --max-time 30 http://127.0.0.1:6333/readyz
curl -fsS http://127.0.0.1:6333/
```

Open the [local dashboard](http://127.0.0.1:6333/dashboard). Only host loopback port
6333 is published. Qdrant uses a named Docker volume at `/qdrant/storage`; it has
no authentication by default and is intended for trusted local use. Telemetry is
disabled. Readiness is checked from the host, without assuming curl exists in the image.

```bash
docker compose logs --tail=50 qdrant
docker compose stop qdrant
docker compose start qdrant
```

Stopping/recreating the container preserves the named volume. Do not use
`docker compose down --volumes` unless you intend to delete stored data.
If port 6333 is already occupied, stop only a service you own or deliberately
change both the Compose host port and `QDRANT_URL`.

During startup the endpoint can return an empty reply, so the readiness command retries
all errors. HNSW measurements belong to Phase 8.
Ollama is not needed for foundation, ingestion, or semantic search. Its native
installation and model preparation belong to Phase 6. `OLLAMA_NO_CLOUD=1` must
be set on that server, not merely in the Python application's `.env`.

## Ingestion

With Qdrant running and the embedding model prepared:

```bash
uv run --locked python scripts/ingest.py
uv run --locked python scripts/inspect_index.py
```

Reingesting unchanged documents embeds and writes nothing. Edited documents are replaced,
with new chunks verified before old ones are deleted. Deleted files stay indexed until you
run `--prune-missing`. The sample collection is too small for an HNSW graph, and inspection
says so. See [vector storage and indexing](docs/vector_indexing.md).

## Semantic search

After ingestion, with Qdrant running. Ollama is not needed:

```bash
uv run --locked python scripts/search.py "Can I work remotely?"
uv run --locked python scripts/search.py "How much vacation time do I get?" --top-k 5 --source annual_leave_policy.md
uv run --locked python scripts/search.py "My laptop stopped charging." --json
```

Results show the score, source, page, chunk, and text, with embedding, vector query, and
total timings. Scores rank passages for one question; they are not confidence, and no
score threshold is applied, so a question with no answer still returns its nearest
passages. Search refuses a collection built with a different model or chunking setting,
because such vectors would give plausible but meaningless results.

## RAG assistant

This needs a running local Ollama and its model. See [RAG explained](docs/rag_explained.md).

```bash
uv run --locked python scripts/ask.py "How many days of annual leave do employees get?"
```

If Ollama is down, the command reports `ollama_unavailable` and still shows the retrieved
passages. Validated citations show that a cited source exists, not that the answer is true.

## Checks

```bash
uv lock --check
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

The default suite collects only `tests/unit`. Unit tests isolate configuration
and block socket access. Integration/model/Ollama/benchmark tests will use explicit
paths and require separately prepared prerequisites; tests never start services
or download assets implicitly.

Application logging uses static event names with counts, timings, categories,
and opaque identifiers. Free-form messages, document text, questions, exception
messages, and arbitrary extra fields are excluded by the JSON formatter.
See [engineering decisions](docs/engineering_decisions.md) for contracts and limitations.

Licensed under [GNU AGPL v3](LICENSE) (`AGPL-3.0-only`), consistent with the chosen
PyMuPDF dependency. Dependencies retain their own licenses.
