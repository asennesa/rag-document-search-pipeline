import os

import fastembed
import pytest
import qdrant_client

from rag_pipeline.config import Settings
from rag_pipeline.errors import DocumentError
from rag_pipeline.ingestion.ingestion_pipeline import prepare_documents


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "docs"
    path.mkdir()
    return path


def run(root, fake_tokenizer, **settings):
    return prepare_documents(Settings(**settings), fake_tokenizer, "tok", root)


def test_failures_are_reported_without_stopping_other_documents(root, fake_tokenizer):
    (root / "good.md").write_text("Useful policy text.")
    (root / "bad.txt").write_bytes(b"caf\xe9")
    (root / "image.png").write_bytes(b"x")
    report = run(root, fake_tokenizer)
    assert [p.document.source for p in report.documents] == ["good.md"]
    assert report.chunk_count == 1
    assert {(f.source, f.category) for f in report.failures} == {
        ("bad.txt", "invalid_encoding"),
        ("image.png", "unsupported_type"),
    }


def test_reports_are_deterministic(root, fake_tokenizer):
    (root / "a.md").write_text("word " * 500)
    first, second = run(root, fake_tokenizer), run(root, fake_tokenizer)
    assert first.documents == second.documents
    assert first.processing_fingerprint == second.processing_fingerprint


def test_editing_a_document_changes_chunks_but_keeps_document_id(root, fake_tokenizer):
    path = root / "a.md"
    path.write_text("Version one.")
    before = run(root, fake_tokenizer).documents[0]
    path.write_text("Version two.")
    after = run(root, fake_tokenizer).documents[0]
    assert before.document.document_id == after.document.document_id
    assert before.revision_id != after.revision_id
    assert before.chunks[0].chunk_id != after.chunks[0].chunk_id


def test_touching_a_file_changes_nothing(root, fake_tokenizer):
    path = root / "a.md"
    path.write_text("Stable.")
    before = run(root, fake_tokenizer).documents[0]
    os.utime(path, (1, 1))
    after = run(root, fake_tokenizer).documents[0]
    assert before.revision_id == after.revision_id
    assert before.chunks == after.chunks


def test_changing_policy_changes_revision_and_chunk_ids(root, fake_tokenizer):
    (root / "a.md").write_text("Stable.")
    before = run(root, fake_tokenizer).documents[0]
    after = run(root, fake_tokenizer, chunk_size=100).documents[0]
    assert before.revision_id != after.revision_id
    assert before.chunks[0].chunk_id != after.chunks[0].chunk_id


def test_symlink_alias_is_reported_once(root, fake_tokenizer):
    (root / "a.md").write_text("Text.")
    (root / "b.md").symlink_to(root / "a.md")
    report = run(root, fake_tokenizer)
    assert len(report.documents) == 1
    assert [f.category for f in report.failures] == ["duplicate_source"]


def test_missing_root_is_a_run_level_error(tmp_path, fake_tokenizer):
    with pytest.raises(DocumentError):
        run(tmp_path / "missing", fake_tokenizer)


def test_preparation_needs_no_embedding_or_vector_clients(root, fake_tokenizer, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not construct embedding or vector clients")

    monkeypatch.setattr(fastembed.TextEmbedding, "__init__", forbidden)
    monkeypatch.setattr(qdrant_client.QdrantClient, "__init__", forbidden)
    (root / "a.md").write_text("Text.")
    assert run(root, fake_tokenizer).chunk_count == 1
