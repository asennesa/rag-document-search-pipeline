import os

import pymupdf
import pytest

from rag_pipeline.errors import DocumentError
from rag_pipeline.identifiers import sha256_bytes
from rag_pipeline.ingestion.document_loader import LoadLimits, discover_documents, load_document

LIMITS = LoadLimits(max_bytes=1_000_000, max_pdf_pages=5, max_characters=10_000)


def pdf_bytes(*pages: str, **save_options) -> bytes:
    """Build a fictional PDF; an empty string gives a page with no text, like a scan."""
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    return pdf.tobytes(**save_options)


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def load(path, root, limits=LIMITS):
    return load_document(path, root, "ns", limits)


def category(path, root, limits=LIMITS):
    with pytest.raises(DocumentError) as caught:
        load(path, root, limits)
    return caught.value.category


@pytest.mark.parametrize("name", ["policy.md", "notes.txt", "UPPER.MD"])
def test_text_success_preserves_identity_and_raw_text(tmp_path, name):
    data = "\ufeffCafé rules  \r\n".encode()
    document = load(write(tmp_path, name, data), tmp_path)
    assert document.source == name and document.filename == name
    assert document.document_hash == sha256_bytes(data)
    assert document.size_bytes == len(data)
    assert document.modified_at.tzinfo is not None
    [page] = document.pages
    assert page.page_number is None
    assert page.raw_text == "Café rules  \r\n"
    assert page.cleaned_text == "Café rules"


def test_pdf_pages_are_one_based_and_preserved(tmp_path):
    document = load(write(tmp_path, "a.pdf", pdf_bytes("First page", "Second page")), tmp_path)
    assert document.file_type == "pdf"
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "Second page" in document.pages[1].raw_text
    assert document.pages[0].cleaned_text == "First page"
    assert document.pages_without_text == ()
    assert document.extractor.startswith("pymupdf-")


def test_partly_scanned_pdf_reports_missing_pages(tmp_path):
    document = load(write(tmp_path, "a.pdf", pdf_bytes("Text", "", "More")), tmp_path)
    assert document.pages_without_text == (2,)


def test_fully_scanned_pdf_is_rejected(tmp_path):
    assert category(write(tmp_path, "a.pdf", pdf_bytes("", "")), tmp_path) == "no_extractable_text"


def test_corrupt_pdf_is_rejected(tmp_path):
    assert category(write(tmp_path, "a.pdf", b"not a pdf"), tmp_path) == "corrupt_pdf"
    truncated = pdf_bytes("Hello")[:40]
    assert category(write(tmp_path, "b.pdf", truncated), tmp_path) == "corrupt_pdf"


def test_encrypted_pdf_is_rejected(tmp_path):
    data = pdf_bytes(
        "Secret", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner"
    )
    assert category(write(tmp_path, "a.pdf", data), tmp_path) == "encrypted_pdf"


def test_pdf_limits_reject_rather_than_index_a_prefix(tmp_path):
    many_pages = write(tmp_path, "pages.pdf", pdf_bytes(*["x"] * 6))
    assert category(many_pages, tmp_path) == "too_many_pages"
    few_characters = LoadLimits(1_000_000, 5, 3)
    long_text = write(tmp_path, "text.pdf", pdf_bytes("abcdefgh"))
    assert category(long_text, tmp_path, few_characters) == "too_many_characters"


@pytest.mark.parametrize("data", [b"", b"  \r\n\t"])
def test_empty_text_files_are_rejected(tmp_path, data):
    assert category(write(tmp_path, "a.txt", data), tmp_path) == "empty_document"


def test_empty_pdf_file_is_rejected(tmp_path):
    assert category(write(tmp_path, "a.pdf", b""), tmp_path) == "empty_document"


def test_unsupported_type_is_rejected_before_reading(tmp_path):
    assert category(tmp_path / "missing.docx", tmp_path) == "unsupported_type"


@pytest.mark.parametrize("data", [b"caf\xe9", "text".encode("utf-16"), b"a\x00b"])
def test_invalid_encoding_is_never_replaced(tmp_path, data):
    assert category(write(tmp_path, "a.txt", data), tmp_path) == "invalid_encoding"


def test_size_and_character_limits(tmp_path):
    big = write(tmp_path, "big.txt", b"x" * 11)
    assert category(big, tmp_path, LoadLimits(10, 5, 100)) == "too_large"
    assert category(big, tmp_path, LoadLimits(100, 5, 10)) == "too_many_characters"


def test_missing_file(tmp_path):
    assert category(tmp_path / "gone.md", tmp_path) == "not_found"


def test_directory_named_like_a_document_is_not_a_file(tmp_path):
    (tmp_path / "folder.md").mkdir()
    assert category(tmp_path / "folder.md", tmp_path) == "not_a_file"


def test_symlink_escaping_root_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    secret = write(tmp_path, "secret.md", b"private")
    link = root / "link.md"
    link.symlink_to(secret)
    assert category(link, root) == "outside_source_root"


def test_parent_traversal_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    write(tmp_path, "secret.md", b"private")
    assert category(root / ".." / "secret.md", root) == "outside_source_root"


def test_file_changing_during_read_is_rejected(tmp_path, monkeypatch):
    path = write(tmp_path, "a.txt", b"before")
    real_open = type(path).open

    def racing_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        os.utime(self, ns=(1, 1))
        return handle

    monkeypatch.setattr(type(path), "open", racing_open)
    assert category(path, tmp_path) == "source_changed"


def test_same_filename_in_different_folders_or_namespaces_stays_distinct(tmp_path):
    a = load(write(tmp_path, "a/policy.md", b"same"), tmp_path)
    b = load(write(tmp_path, "b/policy.md", b"same"), tmp_path)
    assert a.document_hash == b.document_hash
    assert a.document_id != b.document_id
    assert a.source == "a/policy.md"
    other = load_document(tmp_path / "a/policy.md", tmp_path, "other", LIMITS)
    assert other.document_id != a.document_id


def test_identity_is_independent_of_checkout_location(tmp_path):
    first = load(write(tmp_path, "one/doc.md", b"same"), tmp_path / "one")
    second = load(write(tmp_path, "two/doc.md", b"same"), tmp_path / "two")
    assert first.document_id == second.document_id


def test_unicode_path_forms_share_one_identity(tmp_path):
    decomposed = write(tmp_path, "cafe\u0301.md", b"text")
    assert load(decomposed, tmp_path).source == "caf\u00e9.md"


def test_repeated_loads_are_identical(tmp_path):
    path = write(tmp_path, "a.pdf", pdf_bytes("Stable text"))
    assert load(path, tmp_path) == load(path, tmp_path)


def test_discovery_is_sorted_skips_hidden_and_keeps_unsupported(tmp_path):
    for name in ("b.md", "a/c.txt", "z.docx", ".hidden.md", ".git/config.md"):
        write(tmp_path, name, b"x")
    found = [path.relative_to(tmp_path).as_posix() for path in discover_documents(tmp_path)]
    assert found == ["a/c.txt", "b.md", "z.docx"]


def test_discovery_requires_an_existing_root(tmp_path):
    with pytest.raises(DocumentError) as caught:
        discover_documents(tmp_path / "missing")
    assert caught.value.category == "source_root_missing"


def test_discovery_does_not_follow_directory_symlinks(tmp_path):
    outside = tmp_path / "outside"
    write(outside, "secret.md", b"x")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    assert discover_documents(root) == []
