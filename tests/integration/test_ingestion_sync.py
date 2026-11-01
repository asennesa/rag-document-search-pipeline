"""Full ingestion: real embeddings and real Qdrant, on temporary fictional files.

Requires: prepared embedding model, and docker compose up -d qdrant
"""

import uuid

import pytest

from rag_pipeline.assets import embedding_assets_id, resolve_embedding_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import load_embedding_service
from rag_pipeline.errors import VectorStoreError
from rag_pipeline.ingestion.ingestion_pipeline import ingest_documents
from rag_pipeline.tokenization import EmbeddingTokenizer
from rag_pipeline.vector_store.qdrant_store import connect_store

pytestmark = [pytest.mark.integration, pytest.mark.model]

LONG = "\n\n".join(f"Section {i} explains rule number {i} in detail. " * 12 for i in range(6))


@pytest.fixture(scope="module")
def service():
    return load_embedding_service(Settings())


@pytest.fixture(scope="module")
def tokenizer():
    return EmbeddingTokenizer(resolve_embedding_tokenizer(Settings()).tokenizer_file)


class Harness:
    def __init__(self, root, service, tokenizer):
        self.root = root
        self.tokenizer = tokenizer
        self.settings = Settings(
            qdrant_collection=f"test_phase4_{uuid.uuid4().hex}", documents_dir=root
        )
        assert self.settings.qdrant_collection != Settings().qdrant_collection
        self.store = connect_store(self.settings)
        self.embedded = 0
        self.service = service

    def embed(self, texts):
        self.embedded += len(texts)
        return self.service.embed_documents(texts)

    def ingest(self, **kwargs):
        return ingest_documents(
            self.settings,
            self.store,
            self.tokenizer,
            embedding_assets_id(self.settings),
            self.embed,
            **kwargs,
        )

    def statuses(self, report):
        return {result.source: result.status for result in report.results}


@pytest.fixture
def harness(tmp_path, service, tokenizer):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "long.md").write_text(LONG)
    (root / "other.md").write_text("The other policy covers bicycle parking.")
    built = Harness(root, service, tokenizer)
    yield built
    client = built.store._client
    if client.collection_exists(built.settings.qdrant_collection):
        client.delete_collection(built.settings.qdrant_collection)


def test_reingest_preserves_ids_and_embeds_nothing(harness):
    first = harness.ingest()
    assert set(harness.statuses(first).values()) == {"synced"}
    ids = harness.store.point_ids()
    embedded = harness.embedded
    assert embedded == first.preparation.chunk_count == len(ids)

    second = harness.ingest()
    assert set(harness.statuses(second).values()) == {"unchanged"}
    assert harness.embedded == embedded
    assert harness.store.point_ids() == ids


def test_shortened_document_loses_old_chunks_and_old_text(harness):
    harness.ingest()
    other_id = next(
        document_id
        for document_id, source in harness.store.indexed_documents(
            harness.settings.source_namespace
        ).items()
        if source == "other.md"
    )
    namespace = harness.settings.source_namespace
    other_ids = harness.store.chunk_ids(namespace, other_id)
    before = len(harness.store.point_ids())

    (harness.root / "long.md").write_text("Now a single replacement sentence about ferries.")
    report = harness.ingest()
    assert harness.statuses(report) == {"long.md": "synced", "other.md": "unchanged"}
    assert len(harness.store.point_ids()) == 2 < before
    assert harness.store.chunk_ids(namespace, other_id) == other_ids

    query = harness.service.embed_query("Which sentence is about ferries?")
    hits = harness.store.search(query, 5, source="long.md")
    assert [hit.payload.text for hit in hits] == [
        "Now a single replacement sentence about ferries."
    ]


def test_invalid_update_keeps_the_previous_revision_searchable(harness):
    harness.ingest()
    ids = harness.store.point_ids()
    (harness.root / "other.md").write_bytes(b"caf\xe9")
    report = harness.ingest()
    assert report.stale_sources == ("other.md",)
    assert harness.store.point_ids() == ids


def test_prune_removes_a_deleted_file_on_a_real_server(harness):
    harness.ingest()
    (harness.root / "other.md").unlink()
    assert harness.ingest().missing_sources == ("other.md",)
    assert harness.ingest(prune_missing=True).pruned_sources == ("other.md",)
    namespace = harness.settings.source_namespace
    assert list(harness.store.indexed_documents(namespace).values()) == ["long.md"]


def test_changed_chunking_is_refused_rather_than_mixed(harness):
    harness.ingest()
    ids = harness.store.point_ids()
    changed = Settings(
        qdrant_collection=harness.settings.qdrant_collection,
        documents_dir=harness.root,
        chunk_size=100,
    )
    with pytest.raises(VectorStoreError, match="reindex into a new"):
        ingest_documents(
            changed,
            harness.store,
            harness.tokenizer,
            embedding_assets_id(changed),
            harness.embed,
        )
    assert harness.store.point_ids() == ids


def test_interrupted_update_recovers_on_a_real_server(harness, monkeypatch):
    """New chunks written, then the process dies before deleting the old ones."""
    harness.ingest()
    old_ids = set(harness.store.point_ids())
    (harness.root / "long.md").write_text("Replacement text about harbour ferries.")

    def crash(_point_ids):
        raise VectorStoreError("connection lost")

    with monkeypatch.context() as patch:
        patch.setattr(harness.store, "delete_points", crash)
        report = harness.ingest()
    assert report.has_failures
    assert harness.statuses(report)["long.md"] == "failed"
    assert old_ids < set(harness.store.point_ids())  # Old and new coexist, nothing lost.

    embedded = harness.embedded
    recovered = harness.ingest()
    assert harness.statuses(recovered) == {"long.md": "cleaned", "other.md": "unchanged"}
    assert harness.embedded == embedded
    assert len(harness.store.point_ids()) == 2
