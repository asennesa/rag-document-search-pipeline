"""Deterministic hashes and IDs. Nothing here depends on absolute paths or the clock.

Names are JSON-encoded before hashing so field boundaries stay unambiguous:
("ab", "c") and ("a", "bc") must never produce the same identifier.
"""

import hashlib
import json
import uuid

# Fixed forever: changing it changes every document and chunk ID.
_ID_NAMESPACE = uuid.UUID("6f0f4c3e-52a1-5d5e-9a51-3c0a7e1b2d44")


def _stable_uuid(kind: str, *parts: object) -> str:
    name = json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":"))
    return str(uuid.uuid5(_ID_NAMESPACE, name))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def make_document_id(source_namespace: str, source: str) -> str:
    """Identity of a source location; it survives content edits."""
    return _stable_uuid("document", source_namespace, source)


def make_processing_fingerprint(**policy: object) -> str:
    """Hash everything that changes chunk output; a change requires deliberate reindexing."""
    return sha256_text(json.dumps(policy, sort_keys=True, separators=(",", ":")))


def make_revision_id(document_hash: str, processing_fingerprint: str) -> str:
    return _stable_uuid("revision", document_hash, processing_fingerprint)


def make_chunk_id(
    document_id: str,
    revision_id: str,
    page_number: int | None,
    chunk_index: int,
    content_hash: str,
) -> str:
    return _stable_uuid("chunk", document_id, revision_id, page_number, chunk_index, content_hash)
