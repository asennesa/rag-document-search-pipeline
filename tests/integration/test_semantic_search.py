"""Real retrieval over the sample corpus in a disposable collection. No Ollama.

Requires: prepared embedding model, and docker compose up -d qdrant
"""

import os
import subprocess
import sys
import uuid

import pytest

from rag_pipeline.assets import embedding_assets_id, resolve_embedding_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import load_embedding_service
from rag_pipeline.errors import VectorStoreError
from rag_pipeline.ingestion.ingestion_pipeline import ingest_documents
from rag_pipeline.retrieval.semantic_search import load_search_service
from rag_pipeline.tokenization import EmbeddingTokenizer
from rag_pipeline.vector_store.qdrant_store import connect_store

pytestmark = [pytest.mark.integration, pytest.mark.model]

# Paraphrases that deliberately avoid each policy's own wording where possible.
PARAPHRASES = [
    ("How much vacation time do I get?", "annual_leave_policy.md"),
    ("My laptop stopped charging.", "equipment_policy.md"),
    ("Can I do my job from my house?", "remote_work_policy.md"),
    ("How long must my login secret be?", "password_security.md"),
    ("Who pays for dinner when I am away on a work trip?", "travel_expenses.md"),
    ("I woke up with the flu and cannot come in today.", "sick_leave_policy.md"),
    ("May I accept a present from a supplier?", "workplace_conduct.md"),
    ("I clicked a suspicious link in an email, what now?", "incident_reporting.md"),
]


@pytest.fixture(scope="module")
def settings():
    name = f"test_phase5_{uuid.uuid4().hex}"
    return Settings(qdrant_collection=name, ollama_base_url="http://127.0.0.1:1")


@pytest.fixture(scope="module")
def service(settings):
    """Ingest once, then load through the same factory the CLI uses."""
    assert settings.qdrant_collection != Settings().qdrant_collection
    store = connect_store(settings)
    embedder = load_embedding_service(settings)
    tokenizer = EmbeddingTokenizer(resolve_embedding_tokenizer(settings).tokenizer_file)
    report = ingest_documents(
        settings, store, tokenizer, embedding_assets_id(settings), embedder.embed_documents
    )
    assert not report.has_failures
    yield load_search_service(settings)
    store._client.delete_collection(settings.qdrant_collection)


@pytest.mark.parametrize(("question", "expected"), PARAPHRASES)
def test_paraphrases_find_the_expected_policy_in_the_top_five(service, question, expected):
    response = service.search(question, 5)
    ranked = [(round(r.score, 4), r.source, r.chunk_index) for r in response.results]
    print(f"\nRANKING {question!r} expected {expected}: {ranked}")
    assert expected in [result.source for result in response.results]


def test_results_are_ordered_with_correct_metadata(service):
    response = service.search("How many days of annual leave do employees get?", 5)
    scores = [result.score for result in response.results]
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)
    top = response.results[0]
    assert top.source == top.filename == "annual_leave_policy.md"
    assert top.page_number is None
    assert "25 days" in top.text


@pytest.mark.parametrize("source", ["annual_leave_policy.md", "password_security.md"])
def test_source_filter_never_leaks(service, source):
    response = service.search("How much vacation time do I get?", 20, source)
    assert response.results
    assert {result.source for result in response.results} == {source}


def test_unknown_source_returns_no_passages(service):
    assert service.search("Anything", 5, "no_such_file.md").results == ()


def test_search_is_repeatable(service):
    question = "Can I work remotely?"
    first = service.search(question, 5).results
    assert first == service.search(question, 5).results


def test_search_works_with_ollama_unreachable_and_never_imported(service, settings):
    """A fresh process, since other tests in this run may have imported ollama."""
    code = (
        "import sys\n"
        "from rag_pipeline.config import Settings\n"
        "from rag_pipeline.retrieval.semantic_search import load_search_service\n"
        "response = load_search_service(Settings()).search('My laptop stopped charging.', 3)\n"
        "assert response.results\n"
        "assert 'ollama' not in sys.modules\n"
    )
    environment = {
        **os.environ,
        "QDRANT_COLLECTION": settings.qdrant_collection,
        "OLLAMA_BASE_URL": "http://127.0.0.1:1",
    }
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], env=environment, capture_output=True
    )
    assert done.returncode == 0, done.stderr.decode()


def test_missing_collection_is_reported():
    settings = Settings(qdrant_collection=f"test_phase5_missing_{uuid.uuid4().hex}")
    with pytest.raises(VectorStoreError, match="does not exist"):
        load_search_service(settings)
