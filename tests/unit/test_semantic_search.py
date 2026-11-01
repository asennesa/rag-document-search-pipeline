import io
import json
import subprocess
import sys
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import EmbeddingService
from rag_pipeline.errors import EmbeddingError, SearchError, VectorStoreError
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.models import ChunkPayload, SearchHit
from rag_pipeline.retrieval import semantic_search
from rag_pipeline.retrieval.semantic_search import SemanticSearchService, load_search_service
from rag_pipeline.tokenization import QUERY_INSTRUCTION
from rag_pipeline.vector_store.qdrant_store import QdrantStore

VECTOR = [0.5] * 384


def payload(index=0, source="a.md", text="Passage text"):
    return ChunkPayload(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}{index}")),
        document_id=f"doc-{source}",
        source_namespace="ns",
        source=source,
        filename=source,
        page_number=None,
        chunk_index=index,
        text=text,
        content_hash="hash",
        document_hash="dochash",
        revision_id="rev",
        start_char=0,
        end_char=5,
        token_count=4,
        embedding_model="model",
        embedding_assets="assets",
        processing_fingerprint="fp",
    )


class FakeEmbedder:
    def __init__(self):
        self.questions = []

    def embed_query(self, question):
        self.questions.append(question)
        return VECTOR


class FakeStore:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.calls = []

    def search(self, vector, limit, *, namespace=None, source=None, exact=False):
        self.calls.append((vector, limit, namespace, source, exact))
        return self.hits


def make_service(store=None, embedder=None, default_top_k=5, max_top_k=20):
    return SemanticSearchService(
        embedder or FakeEmbedder(),
        store if store is not None else FakeStore(),
        namespace="ns",
        default_top_k=default_top_k,
        max_top_k=max_top_k,
    )


def test_embedded_vector_reaches_the_store_unchanged():
    store = FakeStore()
    embedder = FakeEmbedder()
    make_service(store, embedder).search("  Question?  ", 3, "a.md")
    assert embedder.questions == ["Question?"]
    assert store.calls == [(VECTOR, 3, "ns", "a.md", False)]


def test_default_top_k_comes_from_configuration():
    store = FakeStore()
    assert make_service(store, default_top_k=7).search("Question").top_k == 7
    assert store.calls[0][1] == 7


@pytest.mark.parametrize("top_k", [1, 20])
def test_top_k_boundaries_are_accepted(top_k):
    assert make_service().search("Question", top_k).top_k == top_k


@pytest.mark.parametrize("top_k", [0, -1, 21, 2.5, True, "5"])
def test_invalid_top_k_is_rejected_before_any_work(top_k):
    store = FakeStore()
    embedder = FakeEmbedder()
    with pytest.raises(SearchError, match="from 1 to 20"):
        make_service(store, embedder).search("Question", top_k)
    assert embedder.questions == [] and store.calls == []


@pytest.mark.parametrize("question", ["", "   \n\t"])
def test_blank_question_is_rejected_before_any_work(question):
    embedder = FakeEmbedder()
    with pytest.raises(SearchError, match="empty"):
        make_service(embedder=embedder).search(question)
    assert embedder.questions == []


@pytest.mark.parametrize("source", ["", "   "])
def test_blank_source_filter_is_rejected_not_ignored(source):
    with pytest.raises(SearchError, match="Source filter"):
        make_service().search("Question", source=source)


def test_source_filter_is_normalized_like_stored_sources():
    store = FakeStore()
    make_service(store).search("Question", source=" café.md ")
    assert store.calls[0][3] == "café.md"


def test_empty_results_are_a_valid_response():
    response = make_service().search("Question")
    assert response.results == ()
    assert response.retrieval_time_ms >= 0


def test_results_keep_scores_and_metadata():
    item = payload(3, "b.md", "Exact text")
    [result] = make_service(FakeStore([SearchHit(score=0.1234, payload=item)])).search("Q").results
    assert result.score == 0.1234
    assert (result.source, result.chunk_index, result.text) == ("b.md", 3, "Exact text")
    assert result.chunk_id == item.chunk_id
    assert result.page_number is None


def test_results_are_sorted_by_score_with_stable_ties():
    hits = [
        SearchHit(score=0.5, payload=payload(2, "b.md")),
        SearchHit(score=0.9, payload=payload(0, "z.md")),
        SearchHit(score=0.5, payload=payload(1, "b.md")),
        SearchHit(score=0.5, payload=payload(0, "a.md")),
    ]
    first = make_service(FakeStore(hits)).search("Q").results
    second = make_service(FakeStore(reversed(hits))).search("Q").results
    assert [(r.source, r.chunk_index) for r in first] == [
        ("z.md", 0),
        ("a.md", 0),
        ("b.md", 1),
        ("b.md", 2),
    ]
    assert first == second
    assert [r.score for r in first] == [0.9, 0.5, 0.5, 0.5]


def test_timings_are_measured_separately():
    response = make_service().search("Question")
    assert response.embedding_time_ms >= 0
    assert response.vector_query_time_ms >= 0
    assert response.retrieval_time_ms >= (
        response.embedding_time_ms + response.vector_query_time_ms
    )


def test_store_and_embedding_failures_are_not_swallowed():
    class DownStore(FakeStore):
        def search(self, *args, **kwargs):
            raise VectorStoreError("Qdrant is unreachable")

    class OversizedEmbedder(FakeEmbedder):
        def embed_query(self, question):
            raise EmbeddingError("Question exceeds 512 tokens")

    with pytest.raises(VectorStoreError):
        make_service(DownStore()).search("Question")
    with pytest.raises(EmbeddingError):
        make_service(embedder=OversizedEmbedder()).search("Question")


def test_logs_contain_counts_and_timings_but_no_question_or_text():
    stream = io.StringIO()
    configure_logging(stream=stream)
    store = FakeStore([SearchHit(score=0.5, payload=payload(text="private passage"))])
    make_service(store).search("private question")
    record = json.loads(stream.getvalue())
    assert record["event"] == "search_completed"
    assert record["chunk_count"] == 1
    assert {"embedding_ms", "retrieval_ms", "duration_ms"} <= set(record)
    assert "private" not in stream.getvalue()


def test_real_embedding_service_adds_the_query_prefix_once_in_a_search(fake_tokenizer):
    class Backend:
        queries = []

        def query_embed(self, query, **kwargs):
            self.queries.append(query)
            yield np.ones(384, dtype=np.float32)

    backend = Backend()
    embedder = EmbeddingService(
        backend,
        fake_tokenizer,
        model_name="model",
        dimensions=384,
        max_tokens=40,
        batch_size=2,
    )
    make_service(embedder=embedder).search("Question")
    assert backend.queries == [QUERY_INSTRUCTION + "Question"]
    with pytest.raises(EmbeddingError, match="shorten"):
        make_service(embedder=embedder).search("word " * 100)


class CapturingClient(QdrantClient):
    def __init__(self):
        super().__init__(":memory:")
        self.query_arguments = None
        self.failure = None

    def query_points(self, collection_name, **kwargs):
        self.query_arguments = kwargs
        if self.failure:
            raise self.failure
        return super().query_points(collection_name, **kwargs)


@pytest.fixture
def real_store():
    client = CapturingClient()
    store = QdrantStore(client, Settings(), sleep=lambda _: None)
    store.ensure_collection()
    unit = [0.0] * 384
    unit[0] = 1.0
    other = [0.0] * 384
    other[1] = 1.0
    store.upsert([(payload(0, "a.md"), unit), (payload(0, "b.md"), other)])
    return store, client, unit


def test_store_uses_the_query_api_with_an_explicit_float_vector(real_store):
    store, client, unit = real_store
    make_service(store).search("Question", 2)
    arguments = client.query_arguments
    assert arguments["query"] == VECTOR
    assert all(type(value) is float for value in arguments["query"])
    assert arguments["with_vectors"] is False
    assert arguments["limit"] == 2
    assert arguments["search_params"].exact is False
    assert arguments["search_params"].hnsw_ef == Settings().qdrant_hnsw_ef


def test_filters_translate_to_namespace_and_source_conditions(real_store):
    store, client, _ = real_store
    make_service(store).search("Question", source="a.md")
    conditions = client.query_arguments["query_filter"].must
    assert {(c.key, c.match.value) for c in conditions} == {
        ("source_namespace", "ns"),
        ("source", "a.md"),
    }


def test_source_filter_never_returns_another_source(real_store):
    store, _, _ = real_store
    response = make_service(store).search("Question", 10, source="a.md")
    assert [result.source for result in response.results] == ["a.md"]


def test_unknown_namespace_returns_nothing(real_store):
    store, _, _ = real_store
    service = SemanticSearchService(
        FakeEmbedder(), store, namespace="other", default_top_k=5, max_top_k=20
    )
    assert service.search("Question").results == ()


def test_invalid_stored_payload_surfaces_as_an_error(real_store):
    store, client, unit = real_store
    client.upsert(
        "document_embeddings",
        [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=unit,
                payload={"source_namespace": "ns", "text": "broken"},
            )
        ],
    )
    with pytest.raises(VectorStoreError, match="invalid payload"):
        make_service(store).search("Question", 10)


@pytest.mark.parametrize("cause", [TimeoutError("timed out"), ConnectionError("refused")])
def test_timeouts_and_outages_give_an_actionable_error(real_store, cause):
    store, client, _ = real_store
    client.failure = ResponseHandlingException(cause)
    with pytest.raises(VectorStoreError, match="docker compose up"):
        make_service(store).search("Question")


class FactoryStore:
    def __init__(self, exists=True, foreign=False):
        self.exists = exists
        self.foreign = foreign

    def collection_exists(self):
        return self.exists

    def validate_collection(self):
        pass

    def validate_contents(self, fingerprint):
        if self.foreign:
            raise VectorStoreError("Collection holds points from another model")


@pytest.fixture
def forbid_model(monkeypatch):
    monkeypatch.setattr(semantic_search, "embedding_assets_id", lambda _settings: "assets")

    def forbidden(_settings):
        pytest.fail("The model must not load when the collection is unusable")

    monkeypatch.setattr(semantic_search, "load_embedding_service", forbidden)


def test_missing_collection_tells_the_user_to_ingest(monkeypatch, forbid_model):
    monkeypatch.setattr(semantic_search, "connect_store", lambda _: FactoryStore(exists=False))
    with pytest.raises(VectorStoreError, match="scripts/ingest.py"):
        load_search_service(Settings())


def test_collection_from_another_vector_space_is_refused(monkeypatch, forbid_model):
    monkeypatch.setattr(semantic_search, "connect_store", lambda _: FactoryStore(foreign=True))
    with pytest.raises(VectorStoreError, match="another model"):
        load_search_service(Settings())


def test_importing_search_loads_no_generation_client():
    code = (
        "import sys; import rag_pipeline.retrieval.semantic_search; "
        "sys.exit(1 if 'ollama' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-W", "ignore", "-c", code]).returncode == 0
