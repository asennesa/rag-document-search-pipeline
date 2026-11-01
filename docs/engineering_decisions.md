# Foundation decisions

Phase 1 creates a Python package, configuration/error/logging foundations, tests,
and a single local Qdrant service. It does not implement pipeline stages.

- Python 3.12.16 and uv 0.13.6 are pinned. Direct dependencies and the build backend
  use exact versions; `uv.lock` records the complete resolved dependency graph.
  `uv sync --locked` prevents unnoticed lock updates. Runtime behavior with models
  remains a later-phase acceptance gate even when native imports succeed.
- `src/rag_pipeline` keeps package imports separate from repository-relative files.
  Importing the package does not configure logging, instantiate settings, initialize
  inference, contact services, or create directories.
- Pydantic Settings validates values locally. Credentials use `SecretStr`, are
  excluded from settings serialization/repr, and are forbidden inside service URLs.
  Validation exception strings hide input values. Do not log raw validation
  `errors()` dictionaries, which can contain inputs, or entire configuration objects.
- The selected embedding identity is deliberately constrained to BGE small English
  v1.5, 384 dimensions and 512 input tokens. The chunk limit reserves two special
  tokens; Phase 2 must still verify actual complete encoder inputs using the real
  tokenizer. Configuration cannot establish tokenizer or model compatibility.
- RAG context tokens are a cap. Phase 6 must subtract the actual prompt/template,
  output reserve and safety margin using the generation tokenizer. Foundation
  validation only ensures the output and safety reserves leave some input space.
- Structured logging owns only the `rag_pipeline` logger. Callers use static
  snake_case event names and opaque IDs. The formatter drops unapproved fields,
  nonfinite metrics, free-form messages and exception text. An allowlist cannot
  recognize a secret deliberately placed into an otherwise valid ID/event field;
  callers must follow the logging contract. Third-party logging is not reconfigured.
- Qdrant is pinned to v1.20.0, publishes only localhost REST/dashboard, and persists
  in a named volume. HNSW settings are configuration values for Phase 4, not proof
  of index creation or traversal. Host `/readyz` is the readiness gate.
- FastEmbed requires Hugging Face Hub below 2; use 1.33.0. No model constructors
  are called during foundation verification. Assets and private/generated data
  remain ignored. Model manifests are a Phase 2 deliverable.
- Unit tests are deterministic and network-disabled. Later integration tests are
  explicitly selected and must use disposable collections, never production data.
- GNU AGPL v3 was selected under the user's delegated license decision. No project
  dependencies are substituted and no commercial license is purchased.

## Ingestion decisions (Phase 2)

- Token counting uses a separate tokenizer instance with truncation off. FastEmbed
  enables truncation on its own instance, so counting there could never detect overflow.
- Tokenizer files are pinned to a commit of the ONNX export and verified by SHA-256 on
  every run. Only `prepare_assets.py` uses the network. A pinned manifest entry is
  reused, so upgrading assets is a deliberate act.
- PDF text uses content-stream order (`sort=False`). Geometric sorting can interleave
  columns, and neither option solves reading order in general.
- `source` is the root-relative path rather than the bare filename, so equal filenames in
  different folders stay distinct. Top-level sample documents still show just a filename.
- `upload_max_bytes` limits every loaded file, not only uploads.
- PDF fixtures are built at test time instead of storing binary files in the repository.
- Hashing and IDs live in `identifiers.py` so no stage has to import the chunker to
  compute them.

See [ingestion and chunking](ingestion.md) for behavior and limitations.

## Embedding decisions (Phase 3)

- `TextEmbedding` receives `specific_model_path`, the verified snapshot. This skips
  FastEmbed's download logic, so the pinned bytes are what runs and no network is possible
  on that path. `local_files_only` alone would still pick whatever is cached.
- The ONNX file and tokenizer must share one commit, and every load checks all hashes.
- The processing fingerprint uses `embedding_assets_id`, derived from the tracked manifest.
  Dry runs therefore get the correct fingerprint without having the weights.
- The BGE query instruction is applied by the service because FastEmbed does not add it
  for this model. It is part of the fingerprint: changing it changes query vectors.
- Vectors are validated after conversion to float32, the type Qdrant stores, because a
  value can overflow to infinity in that cast.
- Oversized input is an error. The chunker should make it unreachable for documents, so
  this is a second, independent guard against FastEmbed's silent truncation.

## Storage decisions (Phase 4)

- Completeness compares expected IDs, never counts: IDs derive from content, so equal
  counts can describe different text.
- The embedding model loads lazily, so a no-change run needs no model and takes
  milliseconds.
- Every chunk of an edited document is re-embedded. Reusing unchanged chunks would save
  work but complicates recovery, and chunk IDs include the revision anyway.
- Retries cover connection failures, 429, and 5xx. They are safe because upserts and
  deletes are idempotent with deterministic IDs. Collection creation is not retried.
- `validate_contents` treats points lacking a fingerprint as foreign, so unknown data
  produces an error rather than being mixed or deleted.
- `os.walk` hides unreadable directories by default. Inventory now fails on them, since
  a partial listing would turn into wrong deletions under pruning.
- `--prune-missing` refuses `--documents-dir`, as another folder is not the authority for
  the configured namespace.
- `check_compatibility=False` avoids a network call in the client constructor; versions
  are pinned instead.
- Unit tests use qdrant-client's local mode for adapter behavior only. HNSW, optimizer,
  and persistence evidence comes from the real server.
- pytest uses `--import-mode=importlib`, since unit and integration suites share file
  names and could not otherwise be collected together.

## Retrieval decisions (Phase 5)

- Search validates the collection's fingerprint before answering. Dimensions cannot detect
  a different vector space, and such a query returns plausible nonsense, not an error.
- No score threshold is applied. A single cosine cutoff is not a valid answerability test;
  any threshold must be evaluated against labeled cases.
- Equal scores get a stable order by source and chunk; scores themselves are unchanged.
- Search always filters on the configured source namespace.
- The 256/40 chunk default is the specification's value and is kept. A diagnostic in
  [PLANS.md](../PLANS.md) Section 17 found smaller chunks ranked better on eight questions.
  That sample is too small to tune on, so Phase 8 should decide with the labeled set.

## RAG decisions (Phase 6)

- The prompt adapter is pinned to the template Ollama actually reports. The first version
  was written from the registry's Go template and its hash; the real server uses the Jinja
  template in the GGUF file, so the hash check refused it and real counts were off by one.
  The adapter now matches Ollama's prompt count exactly, enforced by an integration test.
- Citations are a required schema field. With only prompt rules, 6 of 11 correct answers
  lacked citations. The IDs remain the model's claim and pass through the same validator.
  Inline `[S1]` is no longer requested, but is still checked if it appears.
- `think=False` is sent explicitly. Ollama lists a `thinking` capability for this model,
  apparently because its template mentions `<think>`; no reasoning output was observed.
- Source text is HTML-escaped. Delimiters alone would let a document close its block or
  form `<|im_end|>`. Escaping can only make prompts slightly longer, and counting happens
  after it.
- Failed generation becomes a status, so passages remain visible. Only retrieval failures
  raise.
- `done_reason == "length"` is checked before JSON parsing, since cut-off JSON would
  otherwise look like a malformed reply instead of an incomplete one.
- Refusal uses a schema flag rather than phrase matching.
- The model is verified on first use, so building a service makes no Ollama call and search
  stays independent. The server model is never pulled automatically.
- Unit tests run the real `ollama` client over httpx's mock transport, exercising the wire
  format and error mapping without a server. That is not evidence about model behavior.

## Application decisions (Phase 7)

- `app_services.py` holds all non-UI logic, delegating to existing services. It has no
  Streamlit import, so it is tested directly and the page stays thin.
- `st.cache_resource` holds only the facade, keyed by settings. Questions and answers stay
  in per-session state. Counts are read live, so nothing can go stale after ingestion.
- Models load on first use and failures are not remembered, so the app recovers when Qdrant
  or Ollama returns, without a restart. This was confirmed with a real Qdrant outage.
- Generation is serialized with a lock, because this is one local model on one machine.
- Document and model text is shown only through `st.text` and `st.code`. Those elements show
  literal text, so stored Markdown, HTML, and scripts cannot render.
- Uploads get a separate namespace. In the source namespace, `--prune-missing` would have
  treated them as deleted files. Search covers both namespaces.
- Upload names are generated from the content hash and safe characters. The browser's name
  is never a path, so an upload cannot escape the folder or replace another document.
- Files are validated from a hidden temporary path before being moved into place, so a
  rejected upload leaves nothing behind.

Exact observed versions, infrastructure evidence, commands, and remaining limits
are recorded in the Phase 1 completion record in [PLANS.md](../PLANS.md).
