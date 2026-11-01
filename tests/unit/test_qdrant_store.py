"""Adapter behavior using qdrant-client's local mode, which involves no server.

Local mode is not evidence for HNSW, optimizers, or persistence. Those need the real
server tests in tests/integration.
"""

import json
import math
import uuid

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_pipeline.config import Settings
from rag_pipeline.errors import VectorStoreError
from rag_pipeline.models import ChunkPayload
from rag_pipeline.vector_store.qdrant_store import KEYWORD_FIELDS, QdrantStore

DIMENSIONS = 384


def vector(seed=0):
    values = [0.0] * DIMENSIONS
    values[seed % DIMENSIONS] = 1.0
    return values


def payload(index=0, document="doc", namespace="ns", source="a.md", fingerprint="fp"):
    return ChunkPayload(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}{document}{index}")),
        document_id=document,
        source_namespace=namespace,
        source=source,
        filename=source,
        page_number=None,
        chunk_index=index,
        text=f"Text {index}",
        content_hash="hash",
        document_hash="dochash",
        revision_id="rev",
        start_char=0,
        end_char=5,
        token_count=4,
        embedding_model="model",
        embedding_assets="assets",
        processing_fingerprint=fingerprint,
    )


class RecordingClient(QdrantClient):
    """A local client that can fail or record, as the network layer would."""

    def __init__(self):
        super().__init__(":memory:")
        self.upsert_sizes = []
        self.waits = []
        self.failures = []

    def upsert(self, collection_name, points, wait=True, **kwargs):
        if self.failures:
            raise self.failures.pop(0)
        self.upsert_sizes.append(len(points))
        self.waits.append(wait)
        return super().upsert(collection_name, points, wait=wait, **kwargs)


@pytest.fixture
def client():
    return RecordingClient()


@pytest.fixture
def sleeps():
    return []


def make_store(client, sleeps=None, **settings):
    sink = [] if sleeps is None else sleeps
    return QdrantStore(client, Settings(**settings), sleep=sink.append)


@pytest.fixture
def store(client, sleeps):
    built = make_store(client, sleeps)
    built.ensure_collection()
    return built


def connection_error():
    return ResponseHandlingException(ConnectionError("refused"))


def server_error(status=503):
    return UnexpectedResponse(status, "error", b"secret detail", None)


def test_collection_is_created_with_the_required_contract(client, store):
    info = client.get_collection("document_embeddings")
    assert info.config.params.vectors.size == DIMENSIONS
    assert info.config.params.vectors.distance == models.Distance.COSINE
    assert store.collection_exists()


def test_ensure_collection_is_repeatable(client, store):
    store.upsert([(payload(), vector())])
    store.ensure_collection()
    assert client.count("document_embeddings").count == 1


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (models.VectorParams(size=383, distance=models.Distance.COSINE), "383 dimensions"),
        (models.VectorParams(size=DIMENSIONS, distance=models.Distance.DOT), "cosine"),
        (
            {"named": models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE)},
            "unnamed",
        ),
    ],
)
def test_incompatible_collection_is_rejected_without_mutation(client, config, message):
    client.create_collection("document_embeddings", vectors_config=config)
    before = client.get_collection("document_embeddings")
    with pytest.raises(VectorStoreError, match=message):
        make_store(client).ensure_collection()
    assert client.collection_exists("document_embeddings")
    assert client.get_collection("document_embeddings").payload_schema == before.payload_schema
    assert before.payload_schema == {}


def test_contents_from_another_configuration_are_refused(store):
    store.upsert([(payload(fingerprint="old"), vector())])
    with pytest.raises(VectorStoreError, match="1 points from another"):
        store.validate_contents("new")
    store.validate_contents("old")


def test_points_without_a_fingerprint_count_as_foreign(client, store):
    client.upsert(
        "document_embeddings",
        [models.PointStruct(id=str(uuid.uuid4()), vector=vector(), payload={"source": "x"})],
    )
    with pytest.raises(VectorStoreError, match="another model"):
        store.validate_contents("fp")


def test_upsert_is_batched_and_waits_for_acknowledgement(client, sleeps):
    store = make_store(client, sleeps, qdrant_upsert_batch_size=2)
    store.ensure_collection()
    store.upsert([(payload(i), vector(i)) for i in range(5)])
    assert client.upsert_sizes == [2, 2, 1]
    assert all(client.waits)


def test_vectors_are_stored_as_vectors_not_in_the_payload(client, store):
    item = payload()
    store.upsert([(item, vector(3))])
    [record] = client.retrieve(
        "document_embeddings", [item.chunk_id], with_payload=True, with_vectors=True
    )
    assert record.vector == pytest.approx(vector(3))
    assert set(record.payload) == set(ChunkPayload.model_fields)
    assert "vector" not in json.dumps(record.payload)


@pytest.mark.parametrize(
    "bad",
    [[0.0] * 383, [0.0] * 385, [math.nan] + [0.0] * 383, [math.inf] + [0.0] * 383],
)
def test_bad_vectors_are_rejected_before_any_write(client, store, bad):
    with pytest.raises(VectorStoreError):
        store.upsert([(payload(0), vector()), (payload(1), bad)])
    assert client.upsert_sizes == []


@pytest.mark.parametrize("failure", [connection_error, server_error, lambda: server_error(429)])
def test_transient_failures_retry_with_backoff(client, store, sleeps, failure):
    client.failures = [failure(), failure()]
    store.upsert([(payload(), vector())])
    assert client.upsert_sizes == [1]
    assert sleeps == [0.2, 0.4]


def test_retries_are_bounded(client, store, sleeps):
    client.failures = [connection_error() for _ in range(5)]
    with pytest.raises(VectorStoreError, match="unreachable"):
        store.upsert([(payload(), vector())])
    assert len(sleeps) == 2
    assert len(client.failures) == 2


def test_client_errors_are_not_retried_or_leaked(client, store, sleeps):
    client.failures = [server_error(400)]
    with pytest.raises(VectorStoreError) as caught:
        store.upsert([(payload(), vector())])
    assert sleeps == []
    assert "secret" not in str(caught.value)


def test_a_failed_batch_leaves_earlier_batches_acknowledged(client, sleeps):
    store = make_store(client, sleeps, qdrant_upsert_batch_size=2, qdrant_retries=1)
    store.ensure_collection()

    class FailSecond:
        calls = 0

    real = client.upsert

    def flaky(collection_name, points, wait=True, **kwargs):
        FailSecond.calls += 1
        if FailSecond.calls == 2:
            raise connection_error()
        return real(collection_name, points, wait=wait, **kwargs)

    client.upsert = flaky
    with pytest.raises(VectorStoreError):
        store.upsert([(payload(i), vector(i)) for i in range(4)])
    assert client.count("document_embeddings").count == 2


def test_connection_failure_on_a_read_is_actionable(client, store, monkeypatch):
    def refuse(*args, **kwargs):
        raise connection_error()

    monkeypatch.setattr(client, "get_collection", refuse)
    with pytest.raises(VectorStoreError, match="docker compose up"):
        store.validate_collection()


def test_listing_paginates_past_one_page(client, store, monkeypatch):
    monkeypatch.setattr("rag_pipeline.vector_store.qdrant_store._SCROLL_PAGE", 3)
    items = [payload(i) for i in range(8)]
    store.upsert([(item, vector(i)) for i, item in enumerate(items)])
    assert store.chunk_ids("ns", "doc") == {item.chunk_id for item in items}
    assert store.point_ids() == sorted(item.chunk_id for item in items)


def test_scopes_do_not_leak_between_documents_or_namespaces(store):
    a = payload(0, "a")
    b = payload(0, "b", source="b.md")
    other = payload(0, "a", namespace="other")
    store.upsert([(a, vector(0)), (b, vector(1)), (other, vector(2))])
    assert store.chunk_ids("ns", "a") == {a.chunk_id}
    assert store.indexed_documents("ns") == {"a": "a.md", "b": "b.md"}


def test_document_delete_is_scoped(client, store):
    a = payload(0, "a")
    b = payload(0, "b", source="b.md")
    other = payload(0, "a", namespace="other")
    store.upsert([(a, vector(0)), (b, vector(1)), (other, vector(2))])
    store.delete_document("ns", "a")
    assert store.chunk_ids("ns", "a") == set()
    assert store.chunk_ids("ns", "b") == {b.chunk_id}
    assert store.chunk_ids("other", "a") == {other.chunk_id}


def test_point_delete_removes_only_named_points(store):
    items = [payload(i) for i in range(3)]
    store.upsert([(item, vector(i)) for i, item in enumerate(items)])
    store.delete_points([items[0].chunk_id])
    assert store.chunk_ids("ns", "doc") == {items[1].chunk_id, items[2].chunk_id}


def test_search_ranks_and_validates_payloads(store):
    store.upsert([(payload(i), vector(i)) for i in range(3)])
    [best, *_] = store.search(vector(1), 3)
    assert best.payload.chunk_index == 1
    assert best.score == pytest.approx(1.0)


def test_search_source_filter_never_leaks(store):
    store.upsert(
        [
            (payload(0, "a", source="a.md"), vector(0)),
            (payload(0, "b", source="b.md"), vector(1)),
        ]
    )
    hits = store.search(vector(1), 5, source="a.md")
    assert [hit.payload.source for hit in hits] == ["a.md"]


def test_search_rejects_wrong_dimensions(store):
    with pytest.raises(VectorStoreError, match="383"):
        store.search([0.0] * 383, 3)


def test_invalid_stored_payload_is_an_error(client, store):
    client.upsert(
        "document_embeddings",
        [models.PointStruct(id=str(uuid.uuid4()), vector=vector(), payload={"text": "x"})],
    )
    with pytest.raises(VectorStoreError, match="invalid payload"):
        store.search(vector(), 1)


def test_missing_collection_lists_no_documents(client):
    assert make_store(client).indexed_documents("ns") == {}


def test_keyword_fields_cover_every_filter_used():
    assert {"source_namespace", "document_id", "source"} <= set(KEYWORD_FIELDS)
