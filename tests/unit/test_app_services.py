import threading

import pytest

from rag_pipeline import app_services
from rag_pipeline.app_services import PipelineFacade, stored_upload_name
from rag_pipeline.config import Settings
from rag_pipeline.errors import AssetError, DocumentError, VectorStoreError, WriterLockError
from rag_pipeline.ingestion.ingestion_pipeline import IngestionReport, prepare_documents
from rag_pipeline.ingestion.synchronization import DocumentResult, writer_lock


class FakeStore:
    def __init__(self):
        self.documents = {}

    def indexed_documents(self, namespace):
        return self.documents.get(namespace, {})

    def collection_exists(self):
        return False


@pytest.fixture
def settings(tmp_path):
    return Settings(
        uploads_dir=tmp_path / "uploads",
        ingestion_lock_dir=tmp_path / "locks",
        upload_max_bytes=1000,
    )


@pytest.fixture
def facade(settings, monkeypatch, fake_tokenizer):
    store = FakeStore()
    monkeypatch.setattr(app_services, "connect_store", lambda _settings: store)
    monkeypatch.setattr(app_services, "embedding_assets_id", lambda _settings: "assets")

    class Asset:
        tokenizer_file = "unused"

    monkeypatch.setattr(app_services, "resolve_embedding_tokenizer", lambda _settings: Asset)
    monkeypatch.setattr(app_services, "EmbeddingTokenizer", lambda _file: fake_tokenizer)
    return PipelineFacade(settings)


@pytest.fixture
def ingested(monkeypatch):
    """Records each ingestion's settings and returns a successful report."""
    calls = []

    def fake_ingest(settings, store, tokenizer, assets, embed):
        calls.append(settings)
        preparation = prepare_documents(settings, tokenizer, assets)
        results = tuple(
            DocumentResult(prepared.document.source, "synced", len(prepared.chunks))
            for prepared in preparation.documents
        )
        return IngestionReport(preparation, results, (), ())

    monkeypatch.setattr(app_services, "ingest_documents", fake_ingest)
    return calls


@pytest.mark.parametrize(
    ("filename", "expected_stem"),
    [
        ("policy.md", "policy"),
        ("../../etc/passwd.md", "passwd"),
        ("..\\..\\windows\\evil.md", "evil"),
        ("/absolute/path/doc.md", "doc"),
        ("my report (final).md", "my_report_final"),
        ("....md", "document"),
        ("日本語.md", "document"),
    ],
)
def test_stored_names_never_carry_a_path(filename, expected_stem):
    name = stored_upload_name(filename, b"data")
    assert "/" not in name and "\\" not in name and not name.startswith(".")
    assert name.endswith(f"-{expected_stem}.md")


def test_extension_is_normalized():
    assert stored_upload_name("REPORT.PDF", b"x").endswith("-REPORT.pdf")


@pytest.mark.parametrize("filename", ["run.sh", "doc.docx", "noextension", "archive.md.zip"])
def test_unsupported_upload_types_are_rejected(filename):
    with pytest.raises(DocumentError) as caught:
        stored_upload_name(filename, b"x")
    assert caught.value.category == "unsupported_type"


def test_equal_content_and_name_share_one_identity_and_others_differ():
    assert stored_upload_name("a.md", b"x") == stored_upload_name("a.md", b"x")
    assert stored_upload_name("a.md", b"x") != stored_upload_name("a.md", b"y")


def test_upload_is_stored_under_a_generated_name_and_ingested(facade, settings, ingested):
    result = facade.ingest_upload("../secret.md", b"Fictional upload text.")
    stored = list(settings.uploads_dir.iterdir())
    assert [path.name for path in stored] == [result.source]
    assert result.status == "synced" and result.chunk_count == 1
    assert ingested[0].documents_dir == settings.uploads_dir


def test_uploads_use_their_own_namespace_so_pruning_cannot_see_them(facade, settings, ingested):
    facade.ingest_upload("a.md", b"Fictional upload text.")
    assert ingested[0].source_namespace == "sample-policies-uploads"
    assert settings.source_namespace == "sample-policies"


def test_oversized_upload_is_rejected_before_writing(facade, settings, ingested):
    with pytest.raises(DocumentError) as caught:
        facade.ingest_upload("a.md", b"x" * 1001)
    assert caught.value.category == "too_large"
    assert not settings.uploads_dir.exists()
    assert ingested == []


@pytest.mark.parametrize(
    ("name", "data", "category"),
    [
        ("bad.txt", b"caf\xe9", "invalid_encoding"),
        ("empty.md", b"", "empty_document"),
        ("fake.pdf", b"not a pdf", "corrupt_pdf"),
    ],
)
def test_invalid_upload_leaves_nothing_behind(facade, settings, ingested, name, data, category):
    with pytest.raises(DocumentError) as caught:
        facade.ingest_upload(name, data)
    assert caught.value.category == category
    assert list(settings.uploads_dir.iterdir()) == []
    assert ingested == []


def test_upload_cannot_replace_another_document(facade, settings, ingested):
    first = facade.ingest_upload("a.md", b"First fictional text.")
    second = facade.ingest_upload("a.md", b"Second fictional text.")
    assert first.source != second.source
    assert len(list(settings.uploads_dir.iterdir())) == 2


def test_uploads_respect_the_writer_lock(facade, settings, ingested):
    with writer_lock(settings.ingestion_lock_dir, settings.qdrant_collection):
        with pytest.raises(WriterLockError):
            facade.ingest_upload("a.md", b"Fictional text.")
    assert ingested == []


def test_sample_ingestion_respects_the_writer_lock(facade, settings, ingested):
    with writer_lock(settings.ingestion_lock_dir, settings.qdrant_collection):
        with pytest.raises(WriterLockError):
            facade.ingest_samples()


def test_sources_merge_namespaces_and_stay_live(facade):
    store = facade._store
    store.documents = {"sample-policies": {"1": "b.md"}, "sample-policies-uploads": {"2": "a.md"}}
    assert facade.sources() == ["a.md", "b.md"]
    store.documents["sample-policies"]["3"] = "c.md"
    assert facade.sources() == ["a.md", "b.md", "c.md"]


def test_missing_collection_overview_is_not_an_error(facade):
    overview = facade.overview()
    assert overview.collection_exists is False
    assert overview.stats is None


def test_failed_model_load_is_retried(facade, monkeypatch):
    attempts = []

    def flaky(_settings):
        attempts.append(1)
        if len(attempts) == 1:
            raise AssetError("not prepared")
        return "embedder"

    monkeypatch.setattr(app_services, "load_embedding_service", flaky)
    with pytest.raises(AssetError):
        facade._embedding_service()
    assert facade._embedding_service() == "embedder"


def test_model_loads_once_across_threads(facade, monkeypatch):
    loads = []
    monkeypatch.setattr(
        app_services, "load_embedding_service", lambda _settings: loads.append(1) or "embedder"
    )
    threads = [threading.Thread(target=facade._embedding_service) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert loads == [1]


def test_search_failure_is_not_cached(facade, monkeypatch):
    monkeypatch.setattr(app_services, "load_embedding_service", lambda _settings: "embedder")
    attempts = []

    class Service:
        def search(self, *args):
            return "response"

    def flaky(_settings, _embedder):
        attempts.append(1)
        if len(attempts) == 1:
            raise VectorStoreError("Collection does not exist")
        return Service()

    monkeypatch.setattr(app_services, "load_search_service", flaky)
    with pytest.raises(VectorStoreError):
        facade.search("Question", 5, None)
    assert facade.search("Question", 5, None) == "response"


def test_generation_runs_one_request_at_a_time(facade, monkeypatch):
    active = []
    peak = []

    class Rag:
        def ask(self, *args):
            active.append(1)
            peak.append(len(active))
            threading.Event().wait(0.02)
            active.pop()
            return "answer"

    facade._search = object()
    facade._embedder = object()
    facade._rag = Rag()
    threads = [threading.Thread(target=facade.ask, args=("Question", 5, None)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(peak) == 1
