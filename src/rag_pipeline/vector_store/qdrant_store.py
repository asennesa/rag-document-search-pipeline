"""Qdrant adapter: explicit vectors in, validated payloads out.

Collections are never recreated or dropped here. An incompatible collection is an error
for a person to resolve, not something to repair automatically.
"""

import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)

from rag_pipeline.config import Settings
from rag_pipeline.errors import VectorStoreError
from rag_pipeline.models import ChunkPayload, SearchHit

logger = logging.getLogger(__name__)
T = TypeVar("T")

KEYWORD_FIELDS = (
    "source_namespace",
    "document_id",
    "source",
    "revision_id",
    "processing_fingerprint",
)
_SCROLL_PAGE = 256
_RETRY_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class CollectionStats:
    """Observed state. Configuration here shows eligibility, never that a query used HNSW."""

    name: str
    status: str
    optimizer_status: str
    exact_points: int
    approximate_points: int | None
    indexed_vectors: int | None
    segments: int | None
    dimensions: int
    distance: str
    hnsw_m: int
    hnsw_ef_construct: int
    full_scan_threshold_kb: int
    indexing_threshold_kb: int | None
    payload_indexes: tuple[str, ...]


def connect_store(settings: Settings) -> "QdrantStore":
    """Create a client. check_compatibility is off because the server version is pinned."""
    api_key = settings.qdrant_api_key
    client = QdrantClient(
        url=str(settings.qdrant_url),
        api_key=api_key.get_secret_value() if api_key else None,
        timeout=math.ceil(settings.qdrant_timeout_seconds),
        check_compatibility=False,
    )
    return QdrantStore(client, settings)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, ResponseHandlingException):
        return True  # Connection failures and timeouts: the outcome is unknown.
    return isinstance(error, UnexpectedResponse) and (
        error.status_code == 429 or error.status_code >= 500
    )


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._dimensions = settings.embedding_dimensions
        self._sleep = sleep

    def _call(self, operation: Callable[[], T], *, retry: bool = True) -> T:
        """Translate client errors. Retries are safe because writes use deterministic IDs."""
        attempts = self._settings.qdrant_retries if retry else 1
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except (ApiException, OSError) as error:
                if attempt < attempts and _is_retryable(error):
                    self._sleep(_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
                    continue
                if isinstance(error, ResponseHandlingException | OSError):
                    message = "Qdrant is unreachable; is it running? Try: docker compose up -d"
                else:
                    message = "Qdrant rejected the request"
                raise VectorStoreError(message) from error
        raise AssertionError("unreachable")

    def collection_exists(self) -> bool:
        return self._call(lambda: self._client.collection_exists(self._collection))

    def ensure_collection(self) -> None:
        """Create if absent; otherwise validate before making any change."""
        if not self.collection_exists():
            settings = self._settings
            self._call(
                lambda: self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=self._dimensions, distance=models.Distance.COSINE
                    ),
                    shard_number=1,
                    hnsw_config=models.HnswConfigDiff(
                        m=settings.qdrant_hnsw_m,
                        ef_construct=settings.qdrant_hnsw_ef_construct,
                        full_scan_threshold=settings.qdrant_full_scan_threshold_kb,
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=settings.qdrant_indexing_threshold_kb
                    ),
                ),
                retry=False,  # Unlike upserts, repeating a creation can conflict.
            )
        self.validate_collection()
        self._ensure_payload_indexes()

    def validate_collection(self) -> None:
        vectors = self._info().config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise VectorStoreError(
                "Collection must use one unnamed dense vector; use a new QDRANT_COLLECTION"
            )
        if vectors.size != self._dimensions:
            raise VectorStoreError(
                f"Collection has {vectors.size} dimensions, expected {self._dimensions}; "
                "use a new QDRANT_COLLECTION"
            )
        if vectors.distance != models.Distance.COSINE:
            raise VectorStoreError(
                "Collection must use cosine distance; use a new QDRANT_COLLECTION"
            )

    def validate_contents(self, processing_fingerprint: str) -> None:
        """Refuse to mix vector spaces. Equal dimensions do not make vectors comparable."""
        foreign = self._count(
            models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="processing_fingerprint",
                        match=models.MatchValue(value=processing_fingerprint),
                    )
                ]
            )
        )
        if foreign:
            raise VectorStoreError(
                f"Collection holds {foreign} points from another model or processing "
                "configuration. Nothing was changed; reindex into a new QDRANT_COLLECTION"
            )

    def _ensure_payload_indexes(self) -> None:
        existing = self._info().payload_schema or {}
        for field in KEYWORD_FIELDS:
            if field not in existing:
                self._call(
                    lambda name=field: self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                        wait=True,
                    )
                )

    def _info(self) -> models.CollectionInfo:
        return self._call(lambda: self._client.get_collection(self._collection))

    def _count(self, count_filter: models.Filter | None = None) -> int:
        result = self._call(
            lambda: self._client.count(
                collection_name=self._collection, count_filter=count_filter, exact=True
            )
        )
        return result.count

    def upsert(self, points: Sequence[tuple[ChunkPayload, list[float]]]) -> None:
        """Write in bounded batches, each acknowledged before the next is sent."""
        structs = []
        for payload, vector in points:
            if len(vector) != self._dimensions:
                raise VectorStoreError(f"Vector for chunk has {len(vector)} dimensions")
            if not all(math.isfinite(value) for value in vector):
                raise VectorStoreError("Vector for chunk has non-finite values")
            structs.append(
                models.PointStruct(
                    id=payload.chunk_id, vector=vector, payload=payload.model_dump(mode="json")
                )
            )
        size = self._settings.qdrant_upsert_batch_size
        for start in range(0, len(structs), size):
            batch = structs[start : start + size]
            self._call(
                lambda batch=batch: self._client.upsert(
                    collection_name=self._collection, points=batch, wait=True
                )
            )

    def _scroll(
        self, scroll_filter: models.Filter | None, fields: Sequence[str] | bool
    ) -> Iterator[models.Record]:
        offset = None
        while True:
            records, offset = self._call(
                lambda position=offset: self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=_SCROLL_PAGE,
                    offset=position,
                    with_payload=fields,
                    with_vectors=False,
                )
            )
            yield from records
            if offset is None:
                return

    def chunk_ids(self, namespace: str, document_id: str) -> set[str]:
        records = self._scroll(_document_filter(namespace, document_id), False)
        return {str(record.id) for record in records}

    def point_ids(self) -> list[str]:
        """Every point ID, sorted, for before-and-after comparisons."""
        return sorted(str(record.id) for record in self._scroll(None, False))

    def indexed_documents(self, namespace: str) -> dict[str, str]:
        """Map document ID to source for one namespace. Empty if there is no collection."""
        if not self.collection_exists():
            return {}
        namespace_filter = models.Filter(must=[_match("source_namespace", namespace)])
        return {
            record.payload["document_id"]: record.payload["source"]
            for record in self._scroll(namespace_filter, ["document_id", "source"])
        }

    def delete_points(self, point_ids: Sequence[str]) -> None:
        self._call(
            lambda: self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=list(point_ids)),
                wait=True,
            )
        )

    def delete_document(self, namespace: str, document_id: str) -> None:
        self._call(
            lambda: self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=_document_filter(namespace, document_id)
                ),
                wait=True,
            )
        )

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        *,
        namespace: str | None = None,
        source: str | None = None,
        exact: bool = False,
    ) -> list[SearchHit]:
        """Query with an explicit vector. exact=False permits ANN; it does not force it."""
        if len(vector) != self._dimensions:
            raise VectorStoreError(f"Query vector has {len(vector)} dimensions")
        conditions = []
        if namespace is not None:
            conditions.append(_match("source_namespace", namespace))
        if source is not None:
            conditions.append(_match("source", source))
        response = self._call(
            lambda: self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                query_filter=models.Filter(must=conditions) if conditions else None,
                search_params=models.SearchParams(
                    hnsw_ef=self._settings.qdrant_hnsw_ef, exact=exact
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        try:
            return [
                SearchHit(score=point.score, payload=ChunkPayload(**point.payload))
                for point in response.points
            ]
        except (ValidationError, TypeError) as error:
            raise VectorStoreError("Stored point has an invalid payload") from error

    def stats(self) -> CollectionStats:
        info = self._info()
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise VectorStoreError("Collection does not use one unnamed dense vector")
        return CollectionStats(
            name=self._collection,
            status=str(getattr(info.status, "value", info.status)),
            optimizer_status=str(getattr(info.optimizer_status, "value", info.optimizer_status)),
            exact_points=self._count(),
            approximate_points=info.points_count,
            indexed_vectors=info.indexed_vectors_count,
            segments=info.segments_count,
            dimensions=vectors.size,
            distance=str(getattr(vectors.distance, "value", vectors.distance)),
            hnsw_m=info.config.hnsw_config.m,
            hnsw_ef_construct=info.config.hnsw_config.ef_construct,
            full_scan_threshold_kb=info.config.hnsw_config.full_scan_threshold,
            indexing_threshold_kb=info.config.optimizer_config.indexing_threshold,
            payload_indexes=tuple(sorted(info.payload_schema or {})),
        )


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _document_filter(namespace: str, document_id: str) -> models.Filter:
    return models.Filter(
        must=[_match("source_namespace", namespace), _match("document_id", document_id)]
    )
