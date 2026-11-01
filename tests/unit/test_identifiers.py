import pytest

from rag_pipeline.config import Settings
from rag_pipeline.identifiers import (
    make_chunk_id,
    make_document_id,
    make_processing_fingerprint,
    make_revision_id,
    sha256_bytes,
    sha256_text,
)
from rag_pipeline.ingestion.ingestion_pipeline import processing_fingerprint


def test_known_hashes_and_stable_ids():
    assert sha256_bytes(b"") == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert sha256_text("é") == sha256_bytes("é".encode())
    # Pinned so an accidental namespace or encoding change fails loudly.
    assert make_document_id("ns", "a.md") == "4546f785-5887-5b0f-90d3-8bfc59d697e5"


def test_document_id_depends_on_namespace_and_path_only():
    base = make_document_id("ns", "a.md")
    assert base != make_document_id("other", "a.md")
    assert base != make_document_id("ns", "sub/a.md")


def test_field_boundaries_are_unambiguous():
    assert make_document_id("ab", "c") != make_document_id("a", "bc")


@pytest.mark.parametrize(
    "changed",
    [
        ("d2", "r", 1, 0, "h"),
        ("d", "r2", 1, 0, "h"),
        ("d", "r", 2, 0, "h"),
        ("d", "r", None, 0, "h"),
        ("d", "r", 1, 1, "h"),
        ("d", "r", 1, 0, "h2"),
    ],
)
def test_every_chunk_id_input_matters(changed):
    assert make_chunk_id("d", "r", 1, 0, "h") != make_chunk_id(*changed)


def test_revision_changes_with_content_or_policy():
    base = make_revision_id("hash", "policy")
    assert base == make_revision_id("hash", "policy")
    assert base != make_revision_id("hash2", "policy")
    assert base != make_revision_id("hash", "policy2")


def test_fingerprint_ignores_argument_order():
    assert make_processing_fingerprint(a=1, b=2) == make_processing_fingerprint(b=2, a=1)


def test_chunking_policy_and_tokenizer_change_the_fingerprint():
    base = processing_fingerprint(Settings(), "tokenizer")
    assert base == processing_fingerprint(Settings(), "tokenizer")
    assert base != processing_fingerprint(Settings(chunk_size=200), "tokenizer")
    assert base != processing_fingerprint(Settings(chunk_overlap=20), "tokenizer")
    assert base != processing_fingerprint(Settings(), "other-tokenizer")
