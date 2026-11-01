"""The labeled set against the real corpus, and retrieval on a disposable collection.

Requires: prepared embedding model, and docker compose up -d qdrant
"""

import json
import uuid
from pathlib import Path

import pytest

from rag_pipeline.assets import embedding_assets_id, resolve_embedding_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import load_embedding_service
from rag_pipeline.evaluation.dataset import (
    corpus_fingerprint,
    load_questions,
    read_corpus_lock,
    validate_labels,
)
from rag_pipeline.evaluation.retrieval_metrics import evaluate_retrieval
from rag_pipeline.ingestion.ingestion_pipeline import ingest_documents, prepare_documents
from rag_pipeline.retrieval.semantic_search import load_search_service
from rag_pipeline.tokenization import EmbeddingTokenizer
from rag_pipeline.vector_store.qdrant_store import connect_store

pytestmark = [pytest.mark.integration, pytest.mark.model]

QUESTIONS = Path("data/evaluation/questions.jsonl")
LOCK = Path("data/evaluation/corpus_lock.json")


@pytest.fixture(scope="module")
def tokenizer():
    return EmbeddingTokenizer(resolve_embedding_tokenizer(Settings()).tokenizer_file)


def test_committed_labels_match_the_committed_corpus(tokenizer):
    settings = Settings()
    report = prepare_documents(settings, tokenizer, embedding_assets_id(settings))
    documents = [prepared.document for prepared in report.documents]
    validate_labels(load_questions(QUESTIONS), documents, read_corpus_lock(LOCK))
    assert json.loads(LOCK.read_text())["corpus_fingerprint"] == corpus_fingerprint(documents)


def test_retrieval_evaluation_runs_on_a_disposable_collection(tokenizer):
    settings = Settings(qdrant_collection=f"test_phase8_{uuid.uuid4().hex}")
    store = connect_store(settings)
    embedder = load_embedding_service(settings)
    try:
        ingest_documents(
            settings, store, tokenizer, embedding_assets_id(settings), embedder.embed_documents
        )
        questions = load_questions(QUESTIONS)
        report = evaluate_retrieval(load_search_service(settings, embedder), questions, 5)
    finally:
        store._client.delete_collection(settings.qdrant_collection)
    assert report["overall"]["questions"] == 20
    assert report["unanswerable_excluded"] == 5
    assert 0.0 <= report["overall"]["hit_rate"] <= 1.0
