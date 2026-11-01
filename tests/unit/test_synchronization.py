import stat

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.errors import (
    DocumentError,
    EmbeddingError,
    PipelineError,
    VectorStoreError,
    WriterLockError,
)
from rag_pipeline.ingestion.chunker import Chunker
from rag_pipeline.ingestion.document_loader import discover_documents
from rag_pipeline.ingestion.ingestion_pipeline import (
    LazyEmbedder,
    check_prune_allowed,
    find_missing,
    ingest_documents,
    prepare_documents,
)
from rag_pipeline.ingestion.synchronization import (
    PayloadContext,
    build_payload,
    reconcile_document,
    writer_lock,
)
from rag_pipeline.models import Document, Page, PreparedDocument

CONTEXT = PayloadContext("fingerprint", "model", "assets")


class FakeStore:
    """In-memory store that records the order of operations and injects failures."""

    def __init__(self):
        self.points = {}
        self.calls = []
        self.fail = {}
        self.written_before_failure = 0
        self.lose_writes = False

    def _maybe_fail(self, name):
        self.calls.append(name)
        if self.fail.get(name):
            self.fail[name] -= 1
            raise VectorStoreError(f"{name} failed")

    def chunk_ids(self, namespace, document_id):
        self._maybe_fail("chunk_ids")
        return {
            point_id
            for point_id, payload in self.points.items()
            if payload.source_namespace == namespace and payload.document_id == document_id
        }

    def upsert(self, points):
        self.calls.append("upsert")
        if self.fail.get("upsert"):
            self.fail["upsert"] -= 1
            for payload, _ in points[: self.written_before_failure]:
                self.points[payload.chunk_id] = payload
            raise VectorStoreError("upsert failed")
        if not self.lose_writes:
            for payload, _ in points:
                self.points[payload.chunk_id] = payload

    def delete_points(self, point_ids):
        self._maybe_fail("delete_points")
        for point_id in point_ids:
            del self.points[point_id]

    def ensure_collection(self):
        self._maybe_fail("ensure_collection")

    def validate_contents(self, fingerprint):
        self._maybe_fail("validate_contents")

    def indexed_documents(self, namespace):
        return {
            payload.document_id: payload.source
            for payload in self.points.values()
            if payload.source_namespace == namespace
        }

    def delete_document(self, namespace, document_id):
        self._maybe_fail("delete_document")
        for point_id in self.chunk_ids(namespace, document_id):
            del self.points[point_id]


class CountingEmbedder:
    def __init__(self):
        self.calls = 0
        self.error = None

    def __call__(self, texts):
        self.calls += 1
        if self.error:
            raise self.error
        return [[0.1] * 384 for _ in texts]


def prepared_document(fake_tokenizer, text, revision="rev1", document_id="doc", source="a.md"):
    document = Document(
        document_id=document_id,
        source_namespace="ns",
        source=source,
        filename=source,
        file_type="md",
        document_hash=f"hash-{revision}",
        size_bytes=1,
        modified_at="2026-01-01T00:00:00Z",
        extractor="test",
        pages=(Page(raw_text=text, cleaned_text=text),),
    )
    chunker = Chunker(fake_tokenizer, chunk_size=20, chunk_overlap=2, max_input_tokens=512)
    return PreparedDocument(
        document=document,
        revision_id=revision,
        chunks=tuple(chunker.split_document(document, revision)),
    )


def words(count, word="word"):
    return " ".join(f"{word}{i}" for i in range(count))


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def embed():
    return CountingEmbedder()


@pytest.fixture
def old(fake_tokenizer):
    return prepared_document(fake_tokenizer, words(100, "old"), "rev1")


@pytest.fixture
def new(fake_tokenizer):
    return prepared_document(fake_tokenizer, words(30, "new"), "rev2")


def ids(prepared):
    return {chunk.chunk_id for chunk in prepared.chunks}


def test_new_document_is_embedded_and_written(old, store, embed):
    result = reconcile_document(old, CONTEXT, store, embed)
    assert result.status == "synced"
    assert result.upserted == len(old.chunks) and result.deleted == 0
    assert set(store.points) == ids(old)


def test_unchanged_document_needs_no_embedding_or_writes(old, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    store.calls.clear()
    result = reconcile_document(old, CONTEXT, store, embed)
    assert result.status == "unchanged"
    assert embed.calls == 1
    assert store.calls == ["chunk_ids"]


def test_shortened_document_replaces_text_and_removes_obsolete_chunks(old, new, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    result = reconcile_document(new, CONTEXT, store, embed)
    assert result.status == "synced"
    assert result.deleted == len(old.chunks)
    assert set(store.points) == ids(new)


def test_delete_happens_only_after_upsert_and_verification(old, new, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    store.calls.clear()
    reconcile_document(new, CONTEXT, store, embed)
    assert store.calls == ["chunk_ids", "upsert", "chunk_ids", "delete_points"]


def test_other_documents_are_untouched(fake_tokenizer, old, new, store, embed):
    other = prepared_document(fake_tokenizer, words(50, "x"), "o", "other", "b.md")
    reconcile_document(old, CONTEXT, store, embed)
    reconcile_document(other, CONTEXT, store, embed)
    reconcile_document(new, CONTEXT, store, embed)
    assert ids(other) <= set(store.points)


def test_embedding_failure_writes_and_deletes_nothing(old, new, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    embed.error = EmbeddingError("bad vector")
    store.calls.clear()
    result = reconcile_document(new, CONTEXT, store, embed)
    assert (result.status, result.error_category) == ("failed", "embedding_failed")
    assert store.calls == ["chunk_ids"]
    assert set(store.points) == ids(old)


def test_wrong_embedding_count_is_rejected_before_writing(old, store):
    result = reconcile_document(old, CONTEXT, store, lambda texts: [[0.1] * 384])
    assert result.error_category == "embedding_failed"
    assert store.points == {}


def test_partial_upsert_keeps_old_chunks_and_retry_finishes(old, new, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    store.fail["upsert"] = 1
    store.written_before_failure = 1
    failed = reconcile_document(new, CONTEXT, store, embed)
    assert (failed.status, failed.error_category) == ("failed", "store_failed")
    assert ids(old) <= set(store.points)  # Readers see old and partial new chunks.

    retried = reconcile_document(new, CONTEXT, store, embed)
    assert retried.status == "synced"
    assert set(store.points) == ids(new)


def test_uncertain_acknowledgement_is_failure_then_converges(old, new, store, embed):
    """The server applied the write but the reply was lost."""
    reconcile_document(old, CONTEXT, store, embed)
    store.fail["upsert"] = 1
    store.written_before_failure = len(new.chunks)
    assert reconcile_document(new, CONTEXT, store, embed).status == "failed"

    calls_before = embed.calls
    result = reconcile_document(new, CONTEXT, store, embed)
    assert result.status == "cleaned"  # Everything was there, so only cleanup remained.
    assert embed.calls == calls_before
    assert set(store.points) == ids(new)


def test_failed_verification_deletes_nothing(old, new, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    store.lose_writes = True
    store.calls.clear()
    result = reconcile_document(new, CONTEXT, store, embed)
    assert (result.status, result.error_category) == ("failed", "verification_failed")
    assert "delete_points" not in store.calls
    assert set(store.points) == ids(old)


def test_verification_read_failure_deletes_nothing(old, new, store, embed):
    class FailingVerify(FakeStore):
        def chunk_ids(self, namespace, document_id):
            if "upsert" in self.calls:
                raise VectorStoreError("read failed")
            return super().chunk_ids(namespace, document_id)

    failing = FailingVerify()
    reconcile_document(old, CONTEXT, failing, embed)
    failing.calls.clear()
    result = reconcile_document(new, CONTEXT, failing, embed)
    assert (result.status, result.error_category) == ("failed", "verification_failed")
    assert "delete_points" not in failing.calls


def test_failed_cleanup_is_reported_and_finished_later(old, new, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    store.fail["delete_points"] = 1
    failed = reconcile_document(new, CONTEXT, store, embed)
    assert (failed.status, failed.error_category) == ("failed", "cleanup_failed")
    assert ids(new) <= set(store.points)

    calls_before = embed.calls
    assert reconcile_document(new, CONTEXT, store, embed).status == "cleaned"
    assert embed.calls == calls_before
    assert set(store.points) == ids(new)


def test_interrupted_partial_revision_is_removed_by_a_later_revision(fake_tokenizer, store, embed):
    abandoned = prepared_document(fake_tokenizer, words(60, "draft"), "draft")
    final = prepared_document(fake_tokenizer, words(30, "final"), "final")
    store.fail["upsert"] = 1
    store.written_before_failure = 1
    reconcile_document(abandoned, CONTEXT, store, embed)
    assert len(store.points) == 1
    reconcile_document(final, CONTEXT, store, embed)
    assert set(store.points) == ids(final)


def test_counts_alone_do_not_establish_completeness(fake_tokenizer, store, embed):
    first = prepared_document(fake_tokenizer, "alpha beta", "a")
    second = prepared_document(fake_tokenizer, "gamma delta", "b")
    reconcile_document(first, CONTEXT, store, embed)
    assert len(first.chunks) == len(second.chunks)
    assert reconcile_document(second, CONTEXT, store, embed).status == "synced"


def test_document_without_chunks_never_deletes_indexed_chunks(old, store, embed):
    reconcile_document(old, CONTEXT, store, embed)
    empty = PreparedDocument(document=old.document, revision_id="empty", chunks=())
    result = reconcile_document(empty, CONTEXT, store, embed)
    assert (result.status, result.error_category) == ("failed", "no_chunks")
    assert set(store.points) == ids(old)


def test_payload_carries_identity_and_no_vector(old):
    payload = build_payload(old, old.chunks[0], CONTEXT).model_dump()
    assert payload["chunk_id"] == old.chunks[0].chunk_id
    assert payload["embedding_model"] == "model"
    assert payload["processing_fingerprint"] == "fingerprint"
    assert payload["page_number"] is None
    assert "vector" not in payload


def test_writer_lock_rejects_a_second_writer_and_releases(tmp_path):
    with writer_lock(tmp_path / "locks", "collection"):
        with pytest.raises(WriterLockError):
            with writer_lock(tmp_path / "locks", "collection"):
                pass
        with writer_lock(tmp_path / "locks", "other-collection"):
            pass
    with writer_lock(tmp_path / "locks", "collection"):
        pass


def test_writer_lock_is_released_after_an_error(tmp_path):
    with pytest.raises(RuntimeError):
        with writer_lock(tmp_path, "collection"):
            raise RuntimeError
    with writer_lock(tmp_path, "collection"):
        pass


def test_lazy_embedder_loads_once_and_only_when_needed():
    class Service:
        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

    loads = []
    embedder = LazyEmbedder(lambda: loads.append(1) or Service())
    assert loads == []
    embedder(["a"])
    embedder(["b"])
    assert loads == [1]


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "docs"
    path.mkdir()
    (path / "a.md").write_text("Alpha text.")
    (path / "b.md").write_text("Beta text.")
    return path


def ingest(root, fake_tokenizer, store, embed, **kwargs):
    return ingest_documents(Settings(), store, fake_tokenizer, "assets", embed, root, **kwargs)


def test_second_ingest_is_all_unchanged(root, fake_tokenizer, store, embed):
    ingest(root, fake_tokenizer, store, embed)
    report = ingest(root, fake_tokenizer, store, embed)
    assert {result.status for result in report.results} == {"unchanged"}
    assert embed.calls == 2
    assert not report.has_failures


def test_invalid_update_keeps_old_revision_and_reports_stale(root, fake_tokenizer, store, embed):
    ingest(root, fake_tokenizer, store, embed)
    before = dict(store.points)
    (root / "a.md").write_bytes(b"caf\xe9")
    report = ingest(root, fake_tokenizer, store, embed)
    assert report.stale_sources == ("a.md",)
    assert report.has_failures
    assert report.missing_sources == ()
    assert store.points == before


def test_missing_file_is_reported_but_never_silently_deleted(root, fake_tokenizer, store, embed):
    ingest(root, fake_tokenizer, store, embed)
    (root / "a.md").unlink()
    report = ingest(root, fake_tokenizer, store, embed)
    assert report.missing_sources == ("a.md",)
    assert {payload.source for payload in store.points.values()} == {"a.md", "b.md"}


def test_prune_removes_only_missing_documents(root, fake_tokenizer, store, embed):
    ingest(root, fake_tokenizer, store, embed)
    (root / "a.md").unlink()
    report = ingest(root, fake_tokenizer, store, embed, prune_missing=True)
    assert report.pruned_sources == ("a.md",)
    assert report.missing_sources == ()
    assert {payload.source for payload in store.points.values()} == {"b.md"}


def test_prune_ignores_a_file_that_exists_but_cannot_be_parsed(root, fake_tokenizer, store, embed):
    ingest(root, fake_tokenizer, store, embed)
    (root / "a.md").write_bytes(b"caf\xe9")
    report = ingest(root, fake_tokenizer, store, embed, prune_missing=True)
    assert report.pruned_sources == ()
    assert {payload.source for payload in store.points.values()} == {"a.md", "b.md"}


def test_prune_refuses_an_empty_listing(root, fake_tokenizer, store, embed):
    ingest(root, fake_tokenizer, store, embed)
    before = dict(store.points)
    (root / "a.md").unlink()
    (root / "b.md").unlink()
    with pytest.raises(PipelineError, match="Refusing to prune"):
        ingest(root, fake_tokenizer, store, embed, prune_missing=True)
    assert store.points == before


def test_prune_refusal_happens_before_any_store_access(root, fake_tokenizer, store, embed):
    (root / "a.md").unlink()
    (root / "b.md").unlink()
    with pytest.raises(PipelineError):
        ingest(root, fake_tokenizer, store, embed, prune_missing=True)
    assert store.calls == []


def test_missing_root_stops_before_any_store_access(tmp_path, fake_tokenizer, store, embed):
    with pytest.raises(DocumentError):
        ingest(tmp_path / "unmounted", fake_tokenizer, store, embed, prune_missing=True)
    assert store.calls == []


def test_unreadable_folder_stops_inventory_instead_of_being_skipped(root):
    locked = root / "locked"
    locked.mkdir()
    (locked / "c.md").write_text("Hidden by permissions.")
    locked.chmod(0)
    try:
        with pytest.raises(DocumentError) as caught:
            discover_documents(root)
    finally:
        locked.chmod(stat.S_IRWXU)
    assert caught.value.category == "inventory_failed"


def test_find_missing_uses_discovered_ids_not_prepared_ones(root, fake_tokenizer, store, embed):
    ingest(root, fake_tokenizer, store, embed)
    (root / "a.md").write_bytes(b"caf\xe9")
    report = prepare_documents(Settings(), fake_tokenizer, "assets", root)
    assert find_missing(store, Settings().source_namespace, report) == {}


def test_check_prune_allowed_accepts_a_populated_listing(root, fake_tokenizer):
    check_prune_allowed(prepare_documents(Settings(), fake_tokenizer, "assets", root))
