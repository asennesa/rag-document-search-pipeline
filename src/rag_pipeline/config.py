"""Validated configuration; construction never contacts services or loads models."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]
Timeout = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Settings(BaseSettings):
    """Load constructor values, environment, .env, then defaults (in that order).

    Relative paths are relative to the working directory. Run from the repo root.
    Asset identities are validated against a manifest in later phases.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        str_strip_whitespace=True,
    )

    qdrant_url: HttpUrl = HttpUrl("http://127.0.0.1:6333")
    qdrant_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    qdrant_collection: str = Field(default="document_embeddings", pattern=r"^[a-zA-Z0-9_-]+$")
    qdrant_timeout_seconds: Timeout = 10
    qdrant_hnsw_m: int = Field(default=16, ge=2)
    qdrant_hnsw_ef_construct: PositiveInt = 100
    qdrant_hnsw_ef: PositiveInt = 128
    qdrant_full_scan_threshold_kb: PositiveInt = 10000
    qdrant_indexing_threshold_kb: PositiveInt = 20000
    qdrant_upsert_batch_size: PositiveInt = 64
    qdrant_retries: int = Field(default=3, ge=1, le=10)

    embedding_model: Literal["BAAI/bge-small-en-v1.5"] = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = Field(default=384, ge=384, le=384)
    embedding_max_tokens: int = Field(default=512, ge=512, le=512)
    embedding_batch_size: PositiveInt = 32
    embedding_threads: PositiveInt = 2
    embedding_cache_dir: Path = Path(".cache/fastembed")
    generation_cache_dir: Path = Path(".cache/generation")
    model_assets_path: Path = Path("model_assets.json")
    chunk_size: int = Field(default=256, gt=0, le=510)
    chunk_overlap: int = Field(default=40, ge=0)

    search_top_k: PositiveInt = 5
    search_max_top_k: PositiveInt = 20
    ollama_base_url: HttpUrl = HttpUrl("http://127.0.0.1:11434")
    ollama_model: str = Field(default="qwen3:4b-instruct-2507-q4_K_M", min_length=1)
    ollama_timeout_seconds: Timeout = 120
    ollama_num_ctx: PositiveInt = 4096
    rag_max_context_tokens: PositiveInt = 3000
    rag_max_output_tokens: PositiveInt = 512
    rag_context_safety_tokens: PositiveInt = 128
    rag_temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)

    documents_dir: Path = Path("data/sample_docs")
    source_namespace: str = Field(default="sample-policies", pattern=r"^[a-zA-Z0-9_-]+$")
    uploads_dir: Path = Path("data/uploads")
    ingestion_lock_dir: Path = Path(".runtime/locks")
    upload_max_bytes: PositiveInt = 20 * 1024 * 1024
    pdf_max_pages: PositiveInt = 500
    extracted_max_characters: PositiveInt = 2_000_000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @property
    def uploads_namespace(self) -> str:
        """Kept apart from the source namespace, so pruning can never see uploads."""
        return f"{self.source_namespace}-uploads"

    @field_validator("qdrant_api_key", mode="before")
    @classmethod
    def empty_key_is_absent(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("qdrant_url", "ollama_base_url")
    @classmethod
    def validate_service_url(cls, value: HttpUrl) -> HttpUrl:
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("Service URL must not contain credentials, query, or fragment")
        if value.path not in (None, "/"):
            raise ValueError("Service URL must be a base origin without a path")
        return value

    @field_validator(
        "documents_dir",
        "uploads_dir",
        "ingestion_lock_dir",
        "embedding_cache_dir",
        "generation_cache_dir",
        "model_assets_path",
        mode="before",
    )
    @classmethod
    def reject_empty_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Path must not be empty")
        return value

    @model_validator(mode="after")
    def validate_budgets(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.search_top_k > self.search_max_top_k:
            raise ValueError("SEARCH_TOP_K must not exceed SEARCH_MAX_TOP_K")
        if self.qdrant_hnsw_ef_construct < self.qdrant_hnsw_m:
            raise ValueError("QDRANT_HNSW_EF_CONSTRUCT must be at least QDRANT_HNSW_M")
        if self.rag_max_output_tokens + self.rag_context_safety_tokens >= self.ollama_num_ctx:
            raise ValueError("OLLAMA_NUM_CTX must leave space after output and safety reserves")
        return self
