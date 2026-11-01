# Document ingestion and chunking

Phase 2 turns local files into deterministic, token-bounded chunks. It embeds nothing,
writes to no database, and needs no running service.

```text
file bytes -> snapshot + SHA-256 -> extract pages -> clean -> chunk -> stable IDs
```

## Supported input

| Type | Extraction | Page number |
|---|---|---|
| `.pdf` | PyMuPDF, content-stream order | one-based |
| `.md`, `.txt` | strict UTF-8, BOM allowed | none |

Each file is read once, so its hash and its text describe the same bytes. A file that
changes during the read is rejected. Size, PDF page, and character limits reject the whole
file; a prefix is never indexed.

A failed file becomes a reported failure and the others continue. Categories include
`unsupported_type`, `empty_document`, `invalid_encoding`, `corrupt_pdf`, `encrypted_pdf`,
`no_extractable_text`, `outside_source_root`, `too_large`, and `source_changed`.

**OCR is not supported.** An image-only PDF is rejected as `no_extractable_text`. A PDF
with some text-free pages is accepted, lists those pages in `pages_without_text`, and is
flagged as partial by the dry run.

Hidden files and folders are skipped, and symlinked folders are not followed. A symlink
that resolves outside the source root is rejected.

## Cleaning

Cleaning is conservative and idempotent. It normalizes line endings, removes BOMs and
control characters, strips trailing spaces, and collapses runs of blank lines. It does not
change interior spacing, remove headers and footers, or repair hyphenation, because those
heuristics can delete real content. Raw text is kept beside cleaned text in memory.

## Why chunk at all

An embedding represents one passage with one vector, and the model reads at most 512
tokens. Embedding a whole policy would blur every topic into one vector and would lose
everything past the limit. Chunking gives each passage its own searchable vector.

- **Chunk size:** small chunks are precise but can lose surrounding context; large chunks
  keep context but mix topics, which weakens the match and gives the LLM more noise.
- **Overlap:** a fact on a chunk boundary would otherwise be split in two. Repeating the
  last tokens (40 by default) keeps it whole in at least one chunk, at the cost of some
  duplicate storage.
- **Token limits:** FastEmbed truncates long input silently, so anything past the limit
  would never be searchable and nothing would report it. Limits must be enforced before
  embedding.

## How the chunker works

1. Tokenize the cleaned page with an untruncated copy of the model's real tokenizer.
2. Take up to `CHUNK_SIZE` tokens. Within the last half of that window, prefer the latest
   paragraph break, then line or sentence end, then word gap.
3. Slice by **character offsets** into the source. Text is never decoded from tokens,
   because the tokenizer lowercases and strips accents.
4. Retokenize the slice and shrink it if needed. A slice can tokenize differently from the
   same span of the full text, so the estimate alone proves nothing.
5. Step back by the overlap, extended by at most 8 tokens to start on a whole word.

Chunks stay within a page so page citations are unambiguous. Characters that yield no
tokens, such as zero-width spaces, stay inside a chunk. Text that yields no tokens at all
fails the document rather than disappearing.

The tokenizer files come from the exact ONNX export FastEmbed uses, pinned by commit and
SHA-256 in `model_assets.json`. Integration tests compare encoder input lengths with
FastEmbed's own configured tokenizer.

## Identity

| ID | Derived from | Changes when |
|---|---|---|
| `document_id` | namespace, relative path | the file moves |
| `document_hash` | source bytes | any byte changes |
| `revision_id` | document hash, processing fingerprint | content or policy changes |
| `chunk_id` | document, revision, page, ordinal, text hash | any of those change |

The processing fingerprint covers the extractor version, tokenizer hash, chunk settings,
and a schema version. Touching a file without changing it alters no ID. Absolute paths are
never used, so IDs match across checkouts.

## Commands

```bash
uv run --locked python scripts/prepare_assets.py --embedding-tokenizer
uv run --locked python scripts/ingest.py --dry-run
uv run --locked python scripts/ingest.py --dry-run --documents-dir path/to/docs
```

The dry run exits 0 when every file prepared, 1 when any file failed, and 2 when the run
could not start, such as an unprepared tokenizer.

## Known limitations

- Sentences and tables that cross PDF pages are split at the page boundary.
- Multi-column or unusual PDFs can extract in a poor reading order.
- An edited document re-chunks completely; chunk reuse across revisions is out of scope.
- English is the validated language scope for retrieval.
- Offsets refer to cleaned text, not raw PDF coordinates.
