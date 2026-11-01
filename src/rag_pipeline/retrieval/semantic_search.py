"""Question in, ranked passages out. This works with Ollama absent or unreachable."""

import logging
import time
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from rag_pipeline.assets import embedding_assets_id
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import load_embedding_service
from rag_pipeline.errors import SearchError, VectorStoreError
from rag_pipeline.ingestion.ingestion_pipeline import processing_fingerprint
from rag_pipeline.models import RetrievedPassage, SearchHit, SearchResponse
from rag_pipeline.vector_store.qdrant_store import connect_store

logger = logging.getLogger(__name__)


class QueryEmbedder(Protocol):
    def embed_query(self, question: str) -> list[float]: ...


class PassageStore(Protocol):
    def search(
        self,
        vector: Sequence[float],
        limit: int,
        *,
        namespace: str | tuple[str, ...] | None = None,
        source: str | None = None,
        exact: bool = False,
    ) -> list[SearchHit]: ...


class SemanticSearchService:
    def __init__(
        self,
        embedder: QueryEmbedder,
        store: PassageStore,
        *,
        namespace: str | tuple[str, ...],
        default_top_k: int,
        max_top_k: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._namespace = namespace
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k

    def search(
        self, question: str, top_k: int | None = None, source: str | None = None
    ) -> SearchResponse:
        """Embed once and query once. Failures raise; nothing degrades silently."""
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise SearchError("Question is empty")
        limit = self._validated_top_k(top_k)
        source = self._validated_source(source)

        vector = self._embedder.embed_query(question)
        embedded = time.perf_counter()
        hits = self._store.search(vector, limit, namespace=self._namespace, source=source)
        queried = time.perf_counter()

        # Equal scores get a stable order; the scores themselves are never changed.
        results = sorted(
            (_to_passage(hit) for hit in hits),
            key=lambda passage: (-passage.score, passage.source, passage.chunk_index),
        )
        response = SearchResponse(
            query=question,
            top_k=limit,
            source_filter=source,
            results=tuple(results),
            embedding_time_ms=(embedded - started) * 1000,
            vector_query_time_ms=(queried - embedded) * 1000,
            retrieval_time_ms=(time.perf_counter() - started) * 1000,
        )
        logger.info(
            "search_completed",
            extra={
                "chunk_count": len(results),
                "embedding_ms": response.embedding_time_ms,
                "retrieval_ms": response.vector_query_time_ms,
                "duration_ms": response.retrieval_time_ms,
            },
        )
        return response

    def _validated_top_k(self, top_k: int | None) -> int:
        limit = self._default_top_k if top_k is None else top_k
        if type(limit) is not int or not 1 <= limit <= self._max_top_k:
            raise SearchError(f"Top-K must be a whole number from 1 to {self._max_top_k}")
        return limit

    @staticmethod
    def _validated_source(source: str | None) -> str | None:
        if source is None:
            return None
        normalized = unicodedata.normalize("NFC", source.strip())
        if not normalized:
            raise SearchError("Source filter is empty")
        return normalized


def _to_passage(hit: SearchHit) -> RetrievedPassage:
    payload = hit.payload
    return RetrievedPassage(
        chunk_id=payload.chunk_id,
        document_id=payload.document_id,
        source=payload.source,
        filename=payload.filename,
        page_number=payload.page_number,
        chunk_index=payload.chunk_index,
        text=payload.text,
        score=hit.score,
    )


def load_search_service(
    settings: Settings, embedder: QueryEmbedder | None = None
) -> SemanticSearchService:
    """Check the collection first: it fails faster than loading a model, and a query in
    the wrong vector space returns plausible nonsense rather than an error."""
    store = connect_store(settings)
    if not store.collection_exists():
        raise VectorStoreError(
            f"Collection {settings.qdrant_collection} does not exist; "
            "run: uv run --locked python scripts/ingest.py"
        )
    store.validate_collection()
    store.validate_contents(processing_fingerprint(settings, embedding_assets_id(settings)))
    return SemanticSearchService(
        embedder or load_embedding_service(settings),
        store,
        namespace=(settings.source_namespace, settings.uploads_namespace),
        default_top_k=settings.search_top_k,
        max_top_k=settings.search_max_top_k,
    )
