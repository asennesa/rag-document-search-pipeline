# RAG explained

Retrieval-augmented generation answers a question in two steps: find relevant passages,
then have a language model answer from them. RAG is built on top of semantic search, which
stays usable by itself when the model is unavailable.

```text
question -> retrieval -> context builder -> Ollama -> citation validation -> response
```

**RAG differs from fine-tuning.** Fine-tuning changes a model's weights through training.
RAG leaves the model unchanged and supplies facts in each request, so updating knowledge
only means reindexing documents, and answers can point to their sources.

## Context construction

`ContextBuilder` turns ranked passages into a prompt.

1. **Deduplicate.** Text already contained in a better-ranked passage is dropped; its chunk
   ID is kept as provenance.
2. **Assign IDs.** Passages actually sent get request-local IDs such as `S1`.
3. **Budget.** Whole passages are added while the complete prompt fits. A passage is never
   cut, and each omission is reported with its reason.

Embedding tokens and LLM tokens are unrelated, so the prompt is counted with the Qwen
tokenizer over the fully rendered chat template:

```text
prompt limit = OLLAMA_NUM_CTX - RAG_MAX_OUTPUT_TOKENS - RAG_CONTEXT_SAFETY_TOKENS
```

Source blocks are also capped by `RAG_MAX_CONTEXT_TOKENS`. The adapter is pinned to the
hash of the template Ollama reports, and any other template is refused rather than budgeted
by guesswork. An integration test requires our count to equal Ollama's exactly.

Ollama 0.40.13 uses the Jinja template inside the model file, not the Go template blob
published beside it in the registry, and the two render differently. Reading the registry
template alone gave a count that was off by one.

## Trust boundaries

Documents and retrieved text are untrusted data.

- Source text and questions are HTML-escaped, so a document cannot close its `<source>`
  block or form a chat token such as `<|im_end|>`.
- The system prompt says text inside sources is data, not instructions.
- The request offers no tools, so the model has nothing to execute.
- Citations are checked by code, outside the model.

**These controls limit prompt injection; they do not eliminate it.** Escaping stops
structural breakout, but plain persuasive text can still influence a model.

## Generation

`LLMService` uses the official Ollama client with a JSON schema requiring `answer`,
`citations`, and `insufficient_evidence`. Refusal is therefore explicit rather than guessed
from phrasing, and the model must state its sources. Asking for inline citations in the
prompt alone failed in most real answers, even when the answer was correct.
A schema constrains the shape of a reply, not its truth. `RAG_TEMPERATURE=0` improves
repeatability only; it does not make answers correct.

## Citations and what they prove

The validator checks the model's `citations` list, plus any inline `[S1]` or `[S1, S2]`,
against the sources actually sent. A passage that was retrieved but omitted has no ID, so
citing it fails. Unknown IDs are reported and never repaired by guessing, and a substantive
answer without citations fails. The IDs are always the model's claim; the application never
fills them in.

**A valid citation shows the source exists, not that it supports the claim.** Checking
support needs human review, which Phase 8 covers with a rubric.

## Response statuses

| Status | Meaning | Answer shown |
|---|---|---|
| `answered` | Citations passed validation | Yes |
| `insufficient_evidence` | The model reported that sources do not answer it | Yes, as a refusal |
| `no_context` | Nothing usable was retrieved; the model was not called | No |
| `citation_invalid` | Citations failed; do not treat as grounded | Yes, flagged |
| `incomplete` | Output limit reached | No |
| `prompt_too_large` | The question leaves no room for sources | No |
| `ollama_unavailable`, `timeout` | Server not reachable or too slow | No |
| `model_missing` | Model not pulled; it is never pulled automatically | No |
| `model_unverified` | Template or context does not match the verified setup | No |
| `malformed_response` | Reply did not match the schema | No |

Retrieved passages are returned with every status above, and always shown apart from the
generated answer.

## Missing information

Empty retrieval skips the model. Otherwise, semantic search nearly always returns something,
even for an unanswerable question, and no single cosine cutoff separates relevant from
irrelevant across queries. So no threshold is applied: the model is told to refuse, and
that behavior must be measured rather than assumed.

## Setup

Install Ollama from [ollama.com](https://ollama.com) or with `brew install ollama`, then in
a separate terminal:

```bash
OLLAMA_NO_CLOUD=1 ollama serve
```

`OLLAMA_NO_CLOUD` belongs in the server's environment. If the macOS app is running instead,
quit it first or configure this setting for the app. Upgrading Ollama can change the active
template, which `model_unverified` reports until the budget is rechecked. Python and Ollama both run on the
host, so the default `http://127.0.0.1:11434` needs no Docker networking.

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
uv run --locked python scripts/prepare_assets.py --generation-tokenizer
uv run --locked python scripts/ask.py "How many days of annual leave do employees get?"
```

The model is about 2.5 GB on disk and used 3.2 GB when loaded with a 4096-token context,
fully on the GPU, on one Apple Silicon machine. Planning assumes 16 GiB of RAM for the full
demo. That figure is an engineering budget, not a measured minimum.

## Known limitations

- Answer quality depends on retrieval, and rankings at the default chunk size are mediocre.
- Refusal and injection resistance were checked on a few cases, not measured at scale.
- Answers do not stream, and only one request runs at a time.
- Citation validation cannot detect a claim its cited source does not support.
- Deduplication handles repeated text, not paraphrases.
