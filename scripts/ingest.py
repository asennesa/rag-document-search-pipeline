"""Ingest documents into Qdrant, or prepare them with --dry-run.

Exit codes: 0 all succeeded, 1 some documents failed, 2 the run could not proceed.
"""

import argparse
import sys
import unicodedata
from functools import partial
from pathlib import Path

from rag_pipeline.assets import embedding_assets_id, resolve_embedding_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import load_embedding_service
from rag_pipeline.errors import PipelineError
from rag_pipeline.identifiers import make_document_id
from rag_pipeline.ingestion.ingestion_pipeline import (
    IngestionReport,
    LazyEmbedder,
    PreparationReport,
    check_prune_allowed,
    find_missing,
    ingest_documents,
    prepare_documents,
)
from rag_pipeline.ingestion.synchronization import writer_lock
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.tokenization import EmbeddingTokenizer
from rag_pipeline.vector_store.qdrant_store import connect_store


def print_preparation(report: PreparationReport) -> None:
    print(f"Processing fingerprint: {report.processing_fingerprint}")
    for prepared in report.documents:
        document = prepared.document
        largest = max((chunk.token_count for chunk in prepared.chunks), default=0)
        print(
            f"  ok      {document.source}: {len(document.pages)} page(s), "
            f"{len(prepared.chunks)} chunks, largest {largest} tokens"
        )
    print_warnings(report)
    print(
        f"Documents: {len(report.documents)} prepared, {len(report.failures)} failed. "
        f"Chunks: {report.chunk_count}. Duration: {report.duration_ms:.1f} ms."
    )


def print_warnings(report: PreparationReport) -> None:
    for prepared in report.documents:
        document = prepared.document
        if document.pages_without_text:
            pages = ", ".join(str(page) for page in document.pages_without_text)
            print(f"  warning {document.source}: no text on page(s) {pages}; extraction is partial")
    for failure in report.failures:
        print(f"  failed  {failure.source}: [{failure.category}] {failure.message}")


def print_ingestion(report: IngestionReport) -> None:
    for result in report.results:
        line = f"  {result.status:<9} {result.source}"
        if result.upserted or result.deleted:
            line += f": {result.upserted} upserted, {result.deleted} obsolete deleted"
        if result.message:
            line += f": [{result.error_category}] {result.message}"
        print(line)
    print_warnings(report.preparation)
    for source in report.stale_sources:
        print(f"  stale   {source}: update failed; the previous revision stays searchable")
    for source in report.missing_sources:
        print(f"  missing {source}: file is gone but stays indexed; see --prune-missing")
    for source in report.pruned_sources:
        print(f"  pruned  {source}")
    counts = [result.status for result in report.results]
    print(
        f"Documents: {counts.count('synced')} synced, {counts.count('unchanged')} unchanged, "
        f"{counts.count('cleaned')} cleaned, {counts.count('failed')} failed, "
        f"{len(report.preparation.failures)} unreadable."
    )
    print(
        f"Vectors upserted: {sum(result.upserted for result in report.results)}. "
        f"Obsolete deleted: {sum(result.deleted for result in report.results)}. "
        f"Embedding: {sum(result.embedding_ms for result in report.results):.1f} ms. "
        f"Total: {report.duration_ms:.1f} ms."
    )


def dry_run(settings: Settings, args: argparse.Namespace, tokenizer, assets_id: str) -> int:
    report = prepare_documents(settings, tokenizer, assets_id, args.documents_dir)
    print("Dry run: nothing embedded or written.")
    print_preparation(report)
    if args.prune_missing:  # Only this path contacts Qdrant, and it only reads.
        check_prune_allowed(report)
        store = connect_store(settings)
        missing = find_missing(store, settings.source_namespace, report)
        for source in missing.values():
            print(f"  would prune {source}")
        print(f"Would prune: {len(missing)} document(s).")
    return 1 if report.failures else 0


def delete_source(settings: Settings, source: str, dry_run_only: bool) -> int:
    namespace = settings.source_namespace
    document_id = make_document_id(namespace, unicodedata.normalize("NFC", source))
    store = connect_store(settings)
    if document_id not in store.indexed_documents(namespace):
        print(f"Not indexed in namespace {namespace}: {source}", file=sys.stderr)
        return 2
    if dry_run_only:
        print(f"Would delete {len(store.chunk_ids(namespace, document_id))} chunks of {source}")
        return 0
    with writer_lock(settings.ingestion_lock_dir, settings.qdrant_collection):
        count = len(store.chunk_ids(namespace, document_id))
        store.delete_document(namespace, document_id)
    print(f"Deleted {count} chunks of {source}")
    return 0


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.delete_source:
        return delete_source(settings, args.delete_source, args.dry_run)

    asset = resolve_embedding_tokenizer(settings)
    tokenizer = EmbeddingTokenizer(asset.tokenizer_file)
    assets_id = embedding_assets_id(settings)
    if args.dry_run:
        return dry_run(settings, args, tokenizer, assets_id)

    with writer_lock(settings.ingestion_lock_dir, settings.qdrant_collection):
        report = ingest_documents(
            settings,
            connect_store(settings),
            tokenizer,
            assets_id,
            LazyEmbedder(partial(load_embedding_service, settings)),
            args.documents_dir,
            args.prune_missing,
        )
    print_ingestion(report)
    return 1 if report.has_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="make no changes; alone it needs no running services",
    )
    parser.add_argument(
        "--documents-dir", type=Path, help="override the configured source directory"
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="delete indexed documents whose file is gone from the configured directory",
    )
    parser.add_argument(
        "--delete-source", metavar="PATH", help="delete one document by its relative source path"
    )
    args = parser.parse_args()
    if args.prune_missing and args.documents_dir:
        parser.error("--prune-missing only trusts the configured directory, not --documents-dir")
    if args.delete_source and (args.prune_missing or args.documents_dir):
        parser.error("--delete-source cannot be combined with other source options")

    settings = Settings()
    configure_logging(settings.log_level)
    try:
        return run(args, settings)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
