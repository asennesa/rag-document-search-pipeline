"""Real ONNX inference on the prepared local model. No Qdrant or Ollama.

Requires: uv run --locked python scripts/prepare_assets.py --embedding-model
Run from the repository root.
"""

import socket

import numpy as np
import pytest
from fastembed import TextEmbedding

from rag_pipeline.assets import (
    EMBEDDING_EXPORT_REPOSITORY,
    EMBEDDING_MODEL_FILE,
    embedding_assets_id,
    resolve_embedding_tokenizer,
)
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import cosine_similarity, load_embedding_service
from rag_pipeline.errors import EmbeddingError
from rag_pipeline.ingestion.ingestion_pipeline import prepare_documents
from rag_pipeline.tokenization import QUERY_INSTRUCTION, EmbeddingTokenizer

pytestmark = [pytest.mark.integration, pytest.mark.model]


@pytest.fixture(scope="module")
def settings():
    return Settings()


@pytest.fixture(scope="module")
def service(settings):
    return load_embedding_service(settings)


def assert_valid(vector):
    assert len(vector) == 384
    array = np.asarray(vector)
    assert np.isfinite(array).all()
    assert np.linalg.norm(array) == pytest.approx(1.0, abs=1e-5)  # FastEmbed normalizes.


def test_registry_matches_the_pinned_export():
    [description] = [
        model
        for model in TextEmbedding._list_supported_models()
        if model.model == "BAAI/bge-small-en-v1.5"
    ]
    assert description.sources.hf == EMBEDDING_EXPORT_REPOSITORY
    assert description.model_file == EMBEDDING_MODEL_FILE
    assert description.dim == 384


def test_loading_and_inference_use_no_network(settings, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model loading must not use the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    assert_valid(load_embedding_service(settings).embed_query("Offline question"))


def test_documents_and_questions_give_valid_vectors(service):
    assert_valid(service.embed_documents(["Employees receive 25 days of annual leave."])[0])
    assert_valid(service.embed_query("How much vacation do I get?"))


def test_results_are_repeatable(service):
    text = ["Employees receive 25 days of annual leave."]
    assert service.embed_documents(text) == service.embed_documents(text)


def test_batching_preserves_order_and_meaning(service):
    texts = ["Short text.", "A somewhat longer sentence about password rotation rules.", "Pizza."]
    together = np.asarray(service.embed_documents(texts))
    separate = np.asarray([service.embed_documents([text])[0] for text in texts])
    assert np.allclose(together, separate, atol=1e-3)


def test_batches_larger_than_the_batch_size_stay_complete(service, settings):
    texts = [f"Numbered policy sentence {i}." for i in range(settings.embedding_batch_size + 5)]
    vectors = service.embed_documents(texts)
    assert len(vectors) == len(texts)
    assert len({tuple(vector) for vector in vectors}) == len(texts)


def test_query_instruction_really_changes_the_query_vector(service):
    question = "How much vacation do I get?"
    assert service.embed_query(question) != service.embed_documents([question])[0]
    assert (
        service.embed_query(question) == service.embed_documents([QUERY_INSTRUCTION + question])[0]
    )


def test_overlength_input_is_rejected_by_the_real_tokenizer(service):
    with pytest.raises(EmbeddingError, match="truncated"):
        service.embed_documents(["word " * 511])
    assert_valid(service.embed_documents(["word " * 510])[0])


def test_similar_sentences_outrank_unrelated_ones(service):
    laptop, computer, pizza = service.embed_documents(
        [
            "My laptop battery is dead.",
            "My computer cannot hold a charge.",
            "I enjoy eating pizza.",
        ]
    )
    assert cosine_similarity(laptop, computer) > cosine_similarity(laptop, pizza)


def test_every_sample_chunk_embeds(service, settings):
    asset = resolve_embedding_tokenizer(settings)
    report = prepare_documents(
        settings, EmbeddingTokenizer(asset.tokenizer_file), embedding_assets_id(settings)
    )
    texts = [chunk.text for prepared in report.documents for chunk in prepared.chunks]
    vectors = service.embed_documents(texts)
    assert len(vectors) == len(texts) == report.chunk_count
    for vector in vectors:
        assert_valid(vector)
