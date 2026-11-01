"""Reconcile one document with the vector store, safely across partial failures.

Order matters: embed, upsert, verify, and only then delete. This is an idempotent
procedure, not a transaction. During an update, readers can see old and new chunks
together. On any failure the old chunks stay, and rerunning finishes the work because
chunk IDs are deterministic.
"""

import fcntl
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from rag_pipeline.errors import EmbeddingError, PipelineError, WriterLockError
from rag_pipeline.models import Chunk, ChunkPayload, PreparedDocument

logger = logging.getLogger(__name__)

Status = Literal["unchanged", "synced", "cleaned", "failed"]


class ChunkStore(Protocol):
    def chunk_ids(self, namespace: str, document_id: str) -> set[str]: ...

    def upsert(self, points: Sequence[tuple[ChunkPayload, list[float]]]) -> None: ...

    def delete_points(self, point_ids: Sequence[str]) -> None: ...


@dataclass(frozen=True)
class PayloadContext:
    processing_fingerprint: str
    embedding_model: str
    embedding_assets: str


@dataclass(frozen=True)
class DocumentResult:
    source: str
    status: Status
    upserted: int = 0
    deleted: int = 0
    embedding_ms: float = 0.0
    error_category: str | None = None
    message: str | None = None


def build_payload(
    prepared: PreparedDocument, chunk: Chunk, context: PayloadContext
) -> ChunkPayload:
    document = prepared.document
    return ChunkPayload(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_namespace=document.source_namespace,
        source=chunk.source,
        filename=document.filename,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        content_hash=chunk.content_hash,
        document_hash=chunk.document_hash,
        revision_id=chunk.revision_id,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        token_count=chunk.token_count,
        embedding_model=context.embedding_model,
        embedding_assets=context.embedding_assets,
        processing_fingerprint=context.processing_fingerprint,
    )


def reconcile_document(
    prepared: PreparedDocument,
    context: PayloadContext,
    store: ChunkStore,
    embed: Callable[[Sequence[str]], list[list[float]]],
) -> DocumentResult:
    """Bring one document to its expected chunk set. Never raises a PipelineError."""
    document = prepared.document
    namespace = document.source_namespace
    document_id = document.document_id
    source = document.source
    expected = {chunk.chunk_id for chunk in prepared.chunks}
    if not expected:  # Would otherwise classify every indexed chunk as obsolete.
        return _failed(source, "no_chunks", "Document produced no chunks")

    stage = "store_failed"
    upserted = 0
    embedding_ms = 0.0
    try:
        indexed = store.chunk_ids(namespace, document_id)
        if not expected <= indexed:
            stage = "embedding_failed"
            started = time.perf_counter()
            vectors = embed([chunk.text for chunk in prepared.chunks])
            embedding_ms = (time.perf_counter() - started) * 1000
            if len(vectors) != len(prepared.chunks):
                raise EmbeddingError("Embedding count does not match chunk count")

            stage = "store_failed"
            store.upsert(
                [
                    (build_payload(prepared, chunk, context), vector)
                    for chunk, vector in zip(prepared.chunks, vectors, strict=True)
                ]
            )
            upserted = len(prepared.chunks)

            stage = "verification_failed"
            indexed = store.chunk_ids(namespace, document_id)
            if not expected <= indexed:
                return _failed(
                    source,
                    stage,
                    "Written chunks could not be verified; old chunks kept, rerun to retry",
                    upserted,
                    embedding_ms,
                )

        stage = "cleanup_failed"
        obsolete = sorted(indexed - expected)
        if obsolete:
            store.delete_points(obsolete)
    except PipelineError as error:
        logger.warning("document_sync_failed", extra={"error_category": stage})
        return _failed(source, stage, str(error), upserted, embedding_ms)

    if upserted:
        status: Status = "synced"
    elif obsolete:
        status = "cleaned"  # A prior run wrote everything but crashed before deleting.
    else:
        status = "unchanged"
    return DocumentResult(source, status, upserted, len(obsolete), embedding_ms)


def _failed(
    source: str, category: str, message: str, upserted: int = 0, embedding_ms: float = 0.0
) -> DocumentResult:
    return DocumentResult(
        source,
        "failed",
        upserted=upserted,
        embedding_ms=embedding_ms,
        error_category=category,
        message=message,
    )


@contextmanager
def writer_lock(lock_dir: Path, collection: str) -> Iterator[None]:
    """One writer per collection on this host; the OS releases it if the process dies.

    This does not coordinate writers on different machines.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{collection}.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WriterLockError(
                "Another ingestion is writing to this collection; wait for it to finish"
            ) from error
        yield
