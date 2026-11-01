"""Ingestion: prepare files into chunks, then reconcile them with the vector store.

Preparation embeds and writes nothing, so a dry run needs no services. One bad file
becomes a reported failure and never stops the others.
"""

import logging
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_pipeline.config import Settings
from rag_pipeline.errors import DocumentError, PipelineError
from rag_pipeline.identifiers import (
    make_document_id,
    make_processing_fingerprint,
    make_revision_id,
)
from rag_pipeline.ingestion.chunker import Chunker
from rag_pipeline.ingestion.document_loader import (
    PDF_EXTRACTOR,
    TEXT_EXTRACTOR,
    LoadLimits,
    discover_documents,
    load_document,
)
from rag_pipeline.ingestion.synchronization import (
    ChunkStore,
    DocumentResult,
    PayloadContext,
    reconcile_document,
)
from rag_pipeline.models import PreparedDocument
from rag_pipeline.tokenization import QUERY_INSTRUCTION, TextTokenizer

# Increment when cleaning or chunking behavior changes.
PROCESSING_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentFailure:
    source: str
    category: str
    message: str


@dataclass(frozen=True)
class PreparationReport:
    processing_fingerprint: str
    documents: tuple[PreparedDocument, ...]
    failures: tuple[DocumentFailure, ...]
    duration_ms: float
    discovered_document_ids: frozenset[str] = frozenset()

    @property
    def chunk_count(self) -> int:
        return sum(len(prepared.chunks) for prepared in self.documents)


def processing_fingerprint(settings: Settings, embedding_assets: str) -> str:
    return make_processing_fingerprint(
        schema=PROCESSING_SCHEMA_VERSION,
        pdf_extractor=PDF_EXTRACTOR,
        text_extractor=TEXT_EXTRACTOR,
        embedding_assets=embedding_assets,
        query_instruction=QUERY_INSTRUCTION,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def prepare_documents(
    settings: Settings,
    tokenizer: TextTokenizer,
    embedding_assets: str,
    root: Path | None = None,
) -> PreparationReport:
    started = time.perf_counter()
    root = settings.documents_dir if root is None else root
    fingerprint = processing_fingerprint(settings, embedding_assets)
    chunker = Chunker(
        tokenizer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_input_tokens=settings.embedding_max_tokens,
    )
    limits = LoadLimits(
        max_bytes=settings.upload_max_bytes,
        max_pdf_pages=settings.pdf_max_pages,
        max_characters=settings.extracted_max_characters,
    )

    def source_of(path: Path) -> str:
        return unicodedata.normalize("NFC", path.relative_to(root).as_posix())

    paths = discover_documents(root)
    prepared: dict[str, PreparedDocument] = {}
    failures: list[DocumentFailure] = []
    for path in paths:
        source = source_of(path)
        try:
            document = load_document(path, root, settings.source_namespace, limits)
            if document.document_id in prepared:  # A symlink to a file already seen.
                raise DocumentError("duplicate_source", "File is an alias of another document")
            revision_id = make_revision_id(document.document_hash, fingerprint)
            chunks = chunker.split_document(document, revision_id)
        except DocumentError as error:
            failures.append(DocumentFailure(source, error.category, str(error)))
            logger.warning("document_failed", extra={"error_category": error.category})
            continue
        prepared[document.document_id] = PreparedDocument(
            document=document, revision_id=revision_id, chunks=tuple(chunks)
        )

    report = PreparationReport(
        processing_fingerprint=fingerprint,
        documents=tuple(prepared.values()),
        failures=tuple(failures),
        duration_ms=(time.perf_counter() - started) * 1000,
        discovered_document_ids=frozenset(
            make_document_id(settings.source_namespace, source_of(path)) for path in paths
        ),
    )
    logger.info(
        "documents_prepared",
        extra={
            "document_count": len(report.documents),
            "chunk_count": report.chunk_count,
            "failure_count": len(report.failures),
            "duration_ms": report.duration_ms,
        },
    )
    return report


class IngestionStore(ChunkStore, Protocol):
    """Store behavior the full ingestion run needs beyond one document."""

    def ensure_collection(self) -> None: ...

    def validate_contents(self, processing_fingerprint: str) -> None: ...

    def indexed_documents(self, namespace: str) -> dict[str, str]: ...

    def delete_document(self, namespace: str, document_id: str) -> None: ...


@dataclass(frozen=True)
class IngestionReport:
    preparation: PreparationReport
    results: tuple[DocumentResult, ...]
    stale_sources: tuple[str, ...]  # Failed to prepare; the previous revision stays indexed.
    missing_sources: tuple[str, ...]  # Indexed, but no longer on disk.
    pruned_sources: tuple[str, ...] = ()
    duration_ms: float = 0.0
    failed_results: tuple[DocumentResult, ...] = ()

    @property
    def has_failures(self) -> bool:
        return bool(self.preparation.failures or self.failed_results)


class LazyEmbedder:
    """Load the model on first use, so an unchanged re-ingest never loads it."""

    def __init__(self, load: Callable[[], object]) -> None:
        self._load = load
        self._service = None

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if self._service is None:
            self._service = self._load()
        return self._service.embed_documents(texts)


def find_missing(
    store: IngestionStore, namespace: str, report: PreparationReport
) -> dict[str, str]:
    """Indexed documents whose file is gone. A file that failed to parse is not missing."""
    indexed = store.indexed_documents(namespace)
    return {
        document_id: source
        for document_id, source in indexed.items()
        if document_id not in report.discovered_document_ids
    }


def check_prune_allowed(report: PreparationReport) -> None:
    """An empty listing usually means a missing mount, not that everything was deleted."""
    if not report.discovered_document_ids:
        raise PipelineError("Refusing to prune: the source directory listed no documents")


def ingest_documents(
    settings: Settings,
    store: IngestionStore,
    tokenizer: TextTokenizer,
    embedding_assets: str,
    embed: Callable[[Sequence[str]], list[list[float]]],
    root: Path | None = None,
    prune_missing: bool = False,
) -> IngestionReport:
    """Reconcile every document. Callers must hold the writer lock."""
    started = time.perf_counter()
    report = prepare_documents(settings, tokenizer, embedding_assets, root)
    if prune_missing:
        check_prune_allowed(report)
    store.ensure_collection()
    store.validate_contents(report.processing_fingerprint)

    context = PayloadContext(
        processing_fingerprint=report.processing_fingerprint,
        embedding_model=settings.embedding_model,
        embedding_assets=embedding_assets,
    )
    results = tuple(
        reconcile_document(prepared, context, store, embed) for prepared in report.documents
    )

    namespace = settings.source_namespace
    indexed = store.indexed_documents(namespace)
    stale = tuple(
        failure.source
        for failure in report.failures
        if make_document_id(namespace, failure.source) in indexed
    )
    missing = find_missing(store, namespace, report)
    pruned: list[str] = []
    if prune_missing:
        for document_id, source in missing.items():
            store.delete_document(namespace, document_id)
            pruned.append(source)

    failed = tuple(result for result in results if result.status == "failed")
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "ingestion_completed",
        extra={
            "document_count": len(results),
            "chunk_count": report.chunk_count,
            "vector_count": sum(result.upserted for result in results),
            "deleted_count": sum(result.deleted for result in results),
            "embedding_ms": sum(result.embedding_ms for result in results),
            "failure_count": len(report.failures) + len(failed),
            "duration_ms": duration_ms,
        },
    )
    return IngestionReport(
        preparation=report,
        results=results,
        stale_sources=stale,
        missing_sources=tuple(source for source in missing.values() if source not in pruned),
        pruned_sources=tuple(pruned),
        duration_ms=duration_ms,
        failed_results=failed,
    )
