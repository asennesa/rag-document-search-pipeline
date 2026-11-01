"""Safe local file loading. Each file is read once so hash and text describe one snapshot."""

import os
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pymupdf

from rag_pipeline.errors import DocumentError
from rag_pipeline.identifiers import make_document_id, sha256_bytes
from rag_pipeline.ingestion.text_cleaner import clean_text
from rag_pipeline.models import Document, Page

SUPPORTED_SUFFIXES = {".pdf": "pdf", ".md": "md", ".txt": "txt"}
PDF_EXTRACTOR = f"pymupdf-{pymupdf.__version__}"
TEXT_EXTRACTOR = "utf-8-strict"


@dataclass(frozen=True)
class LoadLimits:
    max_bytes: int
    max_pdf_pages: int
    max_characters: int


def discover_documents(root: Path) -> list[Path]:
    """List regular candidates in sorted order; unsupported files stay in for reporting.

    Hidden entries are skipped. Symlinked directories are not followed. An unreadable
    directory raises instead of being skipped: os.walk ignores such errors by default,
    and a partial listing would make present documents look deleted to pruning.
    """
    if not root.is_dir():
        raise DocumentError("source_root_missing", "Source directory does not exist")

    def inventory_failed(error: OSError) -> None:
        raise DocumentError("inventory_failed", "A source directory cannot be read") from error

    found: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root, onerror=inventory_failed):
        subdirectories[:] = sorted(name for name in subdirectories if not name.startswith("."))
        found.extend(
            Path(directory, name) for name in sorted(filenames) if not name.startswith(".")
        )
    return sorted(found)


def load_document(path: Path, root: Path, namespace: str, limits: LoadLimits) -> Document:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentError("unsupported_type", f"Unsupported file type; expected {supported}")

    source_path, source = _resolve_within_root(path, root)
    data, modified_ns = _read_snapshot(source_path, limits.max_bytes)
    if not data:
        raise DocumentError("empty_document", "File is empty")

    file_type = cast(Literal["pdf", "md", "txt"], SUPPORTED_SUFFIXES[suffix])
    if file_type == "pdf":
        pages = _extract_pdf(data, limits)
        extractor = PDF_EXTRACTOR
    else:
        pages = (_extract_text(data, limits),)
        extractor = TEXT_EXTRACTOR

    missing = tuple(p.page_number for p in pages if p.page_number and not p.cleaned_text)
    if not any(page.cleaned_text for page in pages):
        if file_type == "pdf":
            raise DocumentError(
                "no_extractable_text",
                "PDF has no extractable text; scanned PDFs need OCR, which is unsupported",
            )
        raise DocumentError("empty_document", "File contains only whitespace")

    return Document(
        document_id=make_document_id(namespace, source),
        source_namespace=namespace,
        source=source,
        filename=source_path.name,
        file_type=file_type,
        document_hash=sha256_bytes(data),
        size_bytes=len(data),
        modified_at=datetime.fromtimestamp(modified_ns / 1e9, UTC),
        extractor=extractor,
        pages=pages,
        pages_without_text=missing,
    )


def _resolve_within_root(path: Path, root: Path) -> tuple[Path, str]:
    """Return the canonical file and its normalized root-relative identity."""
    try:
        resolved = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise DocumentError("not_found", "File or source directory does not exist") from error
    except OSError as error:
        raise DocumentError("unreadable", "File path cannot be resolved") from error
    if not resolved.is_relative_to(resolved_root):
        raise DocumentError("outside_source_root", "File resolves outside the source directory")
    if not resolved.is_file():
        raise DocumentError("not_a_file", "Path is not a regular file")
    relative = resolved.relative_to(resolved_root).as_posix()
    return resolved, unicodedata.normalize("NFC", relative)


def _read_snapshot(path: Path, max_bytes: int) -> tuple[bytes, int]:
    """Read at most max_bytes + 1 bytes and reject a file modified during the read."""
    try:
        before = path.stat()
        if before.st_size > max_bytes:
            raise DocumentError("too_large", f"File exceeds the {max_bytes} byte limit")
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        after = path.stat()
    except OSError as error:
        raise DocumentError("unreadable", "File cannot be read") from error
    if len(data) > max_bytes:
        raise DocumentError("too_large", f"File exceeds the {max_bytes} byte limit")
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise DocumentError("source_changed", "File changed while being read; retry")
    return data, after.st_mtime_ns


def _extract_text(data: bytes, limits: LoadLimits) -> Page:
    try:
        raw = data.decode("utf-8-sig")  # Strict: undecodable bytes are never replaced.
    except UnicodeDecodeError as error:
        raise DocumentError("invalid_encoding", "File is not valid UTF-8") from error
    if "\x00" in raw:
        raise DocumentError("invalid_encoding", "File contains NUL bytes; is it really text?")
    if len(raw) > limits.max_characters:
        raise DocumentError("too_many_characters", "Extracted text exceeds the character limit")
    return Page(raw_text=raw, cleaned_text=clean_text(raw))


def _extract_pdf(data: bytes, limits: LoadLimits) -> tuple[Page, ...]:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise DocumentError("encrypted_pdf", "PDF is password protected")
            if pdf.page_count > limits.max_pdf_pages:
                raise DocumentError("too_many_pages", "PDF exceeds the page limit")
            pages: list[Page] = []
            characters = 0
            for index in range(pdf.page_count):
                # sort=False keeps content-stream order, which is deterministic and does
                # not interleave columns the way geometric sorting can.
                raw = pdf[index].get_text("text", sort=False)
                characters += len(raw)
                if characters > limits.max_characters:
                    raise DocumentError(
                        "too_many_characters", "Extracted text exceeds the character limit"
                    )
                pages.append(
                    Page(page_number=index + 1, raw_text=raw, cleaned_text=clean_text(raw))
                )
    except DocumentError:
        raise
    except Exception as error:  # MuPDF raises several unrelated types for malformed input.
        raise DocumentError("corrupt_pdf", "PDF cannot be parsed") from error
    if not pages:
        raise DocumentError("no_extractable_text", "PDF has no pages")
    return tuple(pages)
