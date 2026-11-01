"""Real Qdrant server, synthetic vectors, disposable collections. No model needed.

Requires: docker compose up -d qdrant
"""

import random
import uuid

import pytest
from qdrant_client import models

from rag_pipeline.config import Settings
from rag_pipeline.errors import VectorStoreError
from rag_pipeline.models import ChunkPayload
from rag_pipeline.vector_store.qdrant_store import KEYWORD_FIELDS, connect_store

pytestmark = pytest.mark.integration


def vector(seed):
    rng = random.Random(seed)
    values = [rng.gauss(0, 1) for _ in range(384)]
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values]


def payload(index, document="doc", source="a.md", fingerprint="fp", namespace="ns"):
    return ChunkPayload(
        chunk_id=str(uuid.uuid4()),
        document_id=document,
        source_namespace=namespace,
        source=source,
        filename=source,
        page_number=index + 1,
        chunk_index=index,
        text=f"Synthetic text {index}",
        content_hash=f"hash{index}",
        document_hash="dochash",
        revision_id="rev",
        start_char=0,
        end_char=10,
        token_count=5,
        embedding_model="model",
        embedding_assets="assets",
        processing_fingerprint=fingerprint,
    )


@pytest.fixture
def settings():
    name = f"test_phase4_{uuid.uuid4().hex}"
    assert name != Settings().qdrant_collection
    return Settings(qdrant_collection=name, qdrant_upsert_batch_size=50)


@pytest.fixture
def store(settings):
    connected = connect_store(settings)
    connected.ensure_collection()
    yield connected
    connected._client.delete_collection(settings.qdrant_collection)


def test_resolved_configuration_matches_settings(store, settings):
    stats = store.stats()
    assert stats.dimensions == 384
    assert stats.distance == "Cosine"
    assert stats.hnsw_m == settings.qdrant_hnsw_m
    assert stats.hnsw_ef_construct == settings.qdrant_hnsw_ef_construct
    assert stats.full_scan_threshold_kb == settings.qdrant_full_scan_threshold_kb
    assert stats.indexing_threshold_kb == settings.qdrant_indexing_threshold_kb
    assert set(KEYWORD_FIELDS) <= set(stats.payload_indexes)


def test_small_collection_reports_zero_indexed_vectors(store):
    store.upsert([(payload(i), vector(i)) for i in range(10)])
    stats = store.stats()
    assert stats.exact_points == 10
    assert stats.indexed_vectors == 0


def test_multi_batch_upsert_and_pagination_return_exact_ids(store):
    items = [payload(i) for i in range(620)]  # More than two scroll pages.
    store.upsert([(item, vector(i)) for i, item in enumerate(items)])
    assert store.chunk_ids("ns", "doc") == {item.chunk_id for item in items}
    assert store.stats().exact_points == 620


def test_vectors_round_trip_finite_and_stay_out_of_the_payload(store, settings):
    item = payload(0)
    store.upsert([(item, vector(1))])
    [record] = store._client.retrieve(
        settings.qdrant_collection, [item.chunk_id], with_payload=True, with_vectors=True
    )
    assert record.vector == pytest.approx(vector(1), abs=1e-6)
    assert set(record.payload) == set(ChunkPayload.model_fields)


def test_query_api_filters_and_exact_agree_on_the_nearest_point(store):
    a = [(payload(i, "a", "a.md"), vector(i)) for i in range(5)]
    b = [(payload(i, "b", "b.md"), vector(100 + i)) for i in range(5)]
    store.upsert(a + b)
    target = vector(102)
    assert store.search(target, 1)[0].payload.source == "b.md"
    assert store.search(target, 1, exact=True)[0].payload.chunk_id == b[2][0].chunk_id
    assert {hit.payload.source for hit in store.search(target, 10, source="a.md")} == {"a.md"}
    assert store.search(target, 10, namespace="nothing") == []


def test_scoped_delete_leaves_other_documents(store):
    a = payload(0, "a", "a.md")
    b = payload(0, "b", "b.md")
    store.upsert([(a, vector(1)), (b, vector(2))])
    store.delete_document("ns", "a")
    assert store.point_ids() == [b.chunk_id]


def test_incompatible_real_collection_is_untouched(settings):
    connected = connect_store(settings)
    connected._client.create_collection(
        settings.qdrant_collection,
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
    )
    try:
        with pytest.raises(VectorStoreError, match="128 dimensions"):
            connected.ensure_collection()
        info = connected._client.get_collection(settings.qdrant_collection)
        assert info.config.params.vectors.size == 128
        assert info.payload_schema == {}
    finally:
        connected._client.delete_collection(settings.qdrant_collection)


def test_unreachable_server_gives_an_actionable_error():
    settings = Settings(qdrant_url="http://127.0.0.1:1", qdrant_retries=1, qdrant_timeout_seconds=1)
    with pytest.raises(VectorStoreError, match="docker compose up"):
        connect_store(settings).ensure_collection()
