"""What the UI calls. Keeping this free of Streamlit makes it testable and keeps the UI thin.

Every method delegates to an existing pipeline service. Models load on first use, and a
failed load is not remembered, so starting Qdrant or Ollama later just works. Counts are
read live, so they cannot go stale after ingestion.
"""

import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rag_pipeline.assets import embedding_assets_id, resolve_embedding_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import EmbeddingService, load_embedding_service
from rag_pipeline.errors import DocumentError
from rag_pipeline.identifiers import sha256_bytes
from rag_pipeline.ingestion.document_loader import (
    SUPPORTED_SUFFIXES,
    LoadLimits,
    load_document,
)
from rag_pipeline.ingestion.ingestion_pipeline import IngestionReport, ingest_documents
from rag_pipeline.ingestion.synchronization import writer_lock
from rag_pipeline.models import RagResponse, SearchResponse
from rag_pipeline.rag.rag_service import RAGService, load_rag_service
from rag_pipeline.retrieval.semantic_search import SemanticSearchService, load_search_service
from rag_pipeline.tokenization import EmbeddingTokenizer
from rag_pipeline.vector_store.qdrant_store import CollectionStats, connect_store, interpret_index


@dataclass(frozen=True)
class PipelineOverview:
    collection_exists: bool
    stats: CollectionStats | None
    document_count: int
    uploaded_document_count: int
    embedding_model: str
    interpretation: str | None
    hnsw_ef: int


@dataclass(frozen=True)
class UploadResult:
    source: str
    status: str
    chunk_count: int
    report: IngestionReport


def stored_upload_name(filename: str, data: bytes) -> str:
    """A generated name from the content hash and safe characters only.

    The browser's filename is never used as a path, so uploads cannot escape the folder
    or replace a file. Equal content and name map to one document.
    """
    base = PurePosixPath(filename.replace("\\", "/")).name
    suffix = PurePosixPath(base).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentError("unsupported_type", f"Unsupported file type; expected {supported}")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", PurePosixPath(base).stem).strip("_")[:80]
    return f"{sha256_bytes(data)[:12]}-{stem or 'document'}{suffix}"


class PipelineFacade:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = connect_store(settings)
        self._embedder: EmbeddingService | None = None
        self._search: SemanticSearchService | None = None
        self._rag: RAGService | None = None
        self._load_lock = threading.Lock()  # Sessions share these, so load each once.
        self._generation_lock = threading.Lock()  # One Ollama request at a time.

    def _embedding_service(self) -> EmbeddingService:
        with self._load_lock:
            if self._embedder is None:
                self._embedder = load_embedding_service(self._settings)
            return self._embedder

    def _search_service(self) -> SemanticSearchService:
        embedder = self._embedding_service()
        with self._load_lock:
            if self._search is None:
                self._search = load_search_service(self._settings, embedder)
            return self._search

    def search(self, question: str, top_k: int, source: str | None) -> SearchResponse:
        return self._search_service().search(question, top_k, source)

    def ask(self, question: str, top_k: int, source: str | None) -> RagResponse:
        search = self._search_service()
        with self._load_lock:
            if self._rag is None:
                self._rag = load_rag_service(self._settings, search)
            rag = self._rag
        with self._generation_lock:
            return rag.ask(question, top_k, source)

    def sources(self) -> list[str]:
        settings = self._settings
        indexed = {
            **self._store.indexed_documents(settings.source_namespace),
            **self._store.indexed_documents(settings.uploads_namespace),
        }
        return sorted(set(indexed.values()))

    def overview(self) -> PipelineOverview:
        settings = self._settings
        exists = self._store.collection_exists()
        stats = self._store.stats() if exists else None
        return PipelineOverview(
            collection_exists=exists,
            stats=stats,
            document_count=len(self._store.indexed_documents(settings.source_namespace)),
            uploaded_document_count=len(self._store.indexed_documents(settings.uploads_namespace)),
            embedding_model=settings.embedding_model,
            interpretation=interpret_index(stats) if stats else None,
            hnsw_ef=settings.qdrant_hnsw_ef,
        )

    def _ingest(self, settings: Settings) -> IngestionReport:
        asset = resolve_embedding_tokenizer(settings)
        return ingest_documents(
            settings,
            self._store,
            EmbeddingTokenizer(asset.tokenizer_file),
            embedding_assets_id(settings),
            lambda texts: self._embedding_service().embed_documents(texts),
        )

    def ingest_samples(self) -> IngestionReport:
        """Returns only after every document has been reconciled."""
        settings = self._settings
        with writer_lock(settings.ingestion_lock_dir, settings.qdrant_collection):
            return self._ingest(settings)

    def ingest_upload(self, filename: str, data: bytes) -> UploadResult:
        """Validate fully before anything lands in the uploads folder or Qdrant."""
        settings = self._settings
        if len(data) > settings.upload_max_bytes:
            raise DocumentError("too_large", f"File exceeds {settings.upload_max_bytes} bytes")
        name = stored_upload_name(filename, data)
        root = settings.uploads_dir
        root.mkdir(parents=True, exist_ok=True)
        with writer_lock(settings.ingestion_lock_dir, settings.qdrant_collection):
            self._write_validated(root, name, data)
            report = self._ingest(
                settings.model_copy(
                    update={
                        "source_namespace": settings.uploads_namespace,
                        "documents_dir": root,
                    }
                )
            )
        result = next((item for item in report.results if item.source == name), None)
        prepared = next(
            (p for p in report.preparation.documents if p.document.source == name), None
        )
        return UploadResult(
            source=name,
            status=result.status if result else "failed",
            chunk_count=len(prepared.chunks) if prepared else 0,
            report=report,
        )

    def _write_validated(self, root: Path, name: str, data: bytes) -> None:
        """Parse from a hidden temporary file, which discovery skips, then move it in."""
        settings = self._settings
        limits = LoadLimits(
            max_bytes=settings.upload_max_bytes,
            max_pdf_pages=settings.pdf_max_pages,
            max_characters=settings.extracted_max_characters,
        )
        handle = tempfile.NamedTemporaryFile(
            dir=root, prefix=".upload-", suffix=Path(name).suffix, delete=False
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(data)
            load_document(temporary, root, settings.uploads_namespace, limits)
            os.replace(temporary, root / name)
        finally:
            temporary.unlink(missing_ok=True)


def create_facade(settings: Settings) -> PipelineFacade:
    return PipelineFacade(settings)
