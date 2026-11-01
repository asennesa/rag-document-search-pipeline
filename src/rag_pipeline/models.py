"""Typed records passed between pipeline stages."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Page(_Record):
    """One extraction unit. PDF pages are one-based; Markdown and TXT use None."""

    page_number: int | None = Field(default=None, ge=1)
    raw_text: str
    cleaned_text: str


class Document(_Record):
    document_id: str
    source_namespace: str
    source: str  # Normalized POSIX path relative to the source root.
    filename: str
    file_type: Literal["pdf", "md", "txt"]
    document_hash: str  # SHA-256 of the exact source bytes.
    size_bytes: int = Field(ge=0)
    modified_at: datetime
    extractor: str
    pages: tuple[Page, ...]
    pages_without_text: tuple[int, ...] = ()  # Set means extraction is known to be partial.


class Chunk(_Record):
    chunk_id: str
    document_id: str
    source: str
    page_number: int | None
    chunk_index: int = Field(ge=0)  # Document-wide ordinal, stable across pages.
    text: str = Field(min_length=1)
    content_hash: str  # SHA-256 of `text`.
    document_hash: str
    revision_id: str
    start_char: int = Field(ge=0)  # Offsets into the page's *cleaned* text, not raw PDF text.
    end_char: int = Field(gt=0)
    token_count: int = Field(gt=0)  # Complete encoder input, including special tokens.


class PreparedDocument(_Record):
    """A document and its complete expected chunk set, derived before any mutation."""

    document: Document
    revision_id: str
    chunks: tuple[Chunk, ...]


class ChunkPayload(_Record):
    """Exactly what is stored beside a vector. Validated on both write and read."""

    chunk_id: str
    document_id: str
    source_namespace: str
    source: str
    filename: str
    page_number: int | None
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str
    document_hash: str
    revision_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    token_count: int = Field(gt=0)
    embedding_model: str
    embedding_assets: str
    processing_fingerprint: str
    schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be after start_char")
        return self


class SearchHit(_Record):
    score: float
    payload: ChunkPayload


class RetrievedPassage(_Record):
    """One ranked result. `score` orders passages for this query; it is not confidence."""

    chunk_id: str
    document_id: str
    source: str
    filename: str
    page_number: int | None
    chunk_index: int
    text: str
    score: float


class SearchResponse(_Record):
    query: str
    top_k: int
    source_filter: str | None
    results: tuple[RetrievedPassage, ...]
    embedding_time_ms: float
    vector_query_time_ms: float
    retrieval_time_ms: float  # Whole request, including validation and sorting.


class ContextSource(_Record):
    """A passage actually sent to the model, under a request-local ID such as S1."""

    source_id: str
    passage: RetrievedPassage
    duplicate_chunk_ids: tuple[str, ...] = ()  # Same text elsewhere; provenance is kept.


class OmittedPassage(_Record):
    chunk_id: str
    reason: Literal["duplicate", "context_budget"]


class BuiltContext(_Record):
    system_prompt: str
    user_prompt: str
    sources: tuple[ContextSource, ...]
    omitted: tuple[OmittedPassage, ...]
    prompt_tokens: int  # The complete rendered prompt, template included.
    context_tokens: int  # Source blocks only.


class CitationResult(_Record):
    status: Literal["valid", "invalid", "missing", "not_applicable"]
    cited_ids: tuple[str, ...] = ()
    unknown_ids: tuple[str, ...] = ()
    malformed: tuple[str, ...] = ()


RagStatus = Literal[
    "answered",  # Only here are citations validated. This is not proof of correctness.
    "insufficient_evidence",
    "no_context",
    "citation_invalid",
    "incomplete",
    "prompt_too_large",
    "ollama_unavailable",
    "model_missing",
    "model_unverified",
    "timeout",
    "malformed_response",
]


class RagResponse(_Record):
    question: str
    status: RagStatus
    answer: str | None
    message: str | None = None
    citation: CitationResult | None = None
    sources: tuple[ContextSource, ...] = ()
    retrieved: tuple[RetrievedPassage, ...] = ()
    omitted: tuple[OmittedPassage, ...] = ()
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    retrieval_time_ms: float
    context_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float
