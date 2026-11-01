import io
import json
import math

import numpy as np
import pytest

from rag_pipeline import assets
from rag_pipeline.assets import (
    embedding_assets_id,
    prepare_embedding_model,
    resolve_embedding_model,
)
from rag_pipeline.config import Settings
from rag_pipeline.embeddings import embedding_service
from rag_pipeline.embeddings.embedding_service import (
    EmbeddingService,
    cosine_similarity,
    load_embedding_service,
)
from rag_pipeline.errors import AssetError, EmbeddingError
from rag_pipeline.identifiers import sha256_bytes
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.tokenization import QUERY_INSTRUCTION

DIMENSIONS = 384


def unit_vector(seed: int, size: int = DIMENSIONS) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[seed % size] = 1.0
    return vector


class FakeBackend:
    """Echoes position into the vector, so reordering shows up in tests."""

    def __init__(self, passages=None, query=None):
        self.passages = passages
        self.query = query
        self.passage_calls = []
        self.query_calls = []

    def passage_embed(self, texts, **kwargs):
        self.passage_calls.append((list(texts), kwargs))
        if self.passages is not None:
            yield from self.passages
        else:
            for text in texts:
                yield unit_vector(int(text.split()[-1]))

    def query_embed(self, query, **kwargs):
        self.query_calls.append(query)
        yield from self.query if self.query is not None else [unit_vector(7)]


def make_service(fake_tokenizer, backend=None, max_tokens=512, batch_size=32):
    return EmbeddingService(
        backend or FakeBackend(),
        fake_tokenizer,
        model_name="test-model",
        dimensions=DIMENSIONS,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )


def test_empty_input_runs_no_inference(fake_tokenizer):
    backend = FakeBackend()
    assert make_service(fake_tokenizer, backend).embed_documents([]) == []
    assert backend.passage_calls == []


def test_order_and_representation_are_preserved(fake_tokenizer):
    vectors = make_service(fake_tokenizer).embed_documents(["t 3", "t 1", "t 2"])
    assert [vector.index(1.0) for vector in vectors] == [3, 1, 2]
    assert all(type(value) is float for value in vectors[0])
    assert all(len(vector) == DIMENSIONS for vector in vectors)


def test_batch_size_reaches_backend(fake_tokenizer):
    backend = FakeBackend()
    make_service(fake_tokenizer, backend, batch_size=7).embed_documents(["t 1"])
    assert backend.passage_calls[0][1] == {"batch_size": 7}


def test_documents_never_receive_the_query_instruction(fake_tokenizer):
    backend = FakeBackend()
    make_service(fake_tokenizer, backend).embed_documents(["t 1"])
    assert backend.passage_calls[0][0] == ["t 1"]


def test_single_string_is_not_treated_as_a_sequence(fake_tokenizer):
    with pytest.raises(EmbeddingError, match="sequence"):
        make_service(fake_tokenizer).embed_documents("not a list")


@pytest.mark.parametrize("texts", [[""], ["ok 1", "   \n"]])
def test_blank_text_is_rejected_before_inference(fake_tokenizer, texts):
    backend = FakeBackend()
    with pytest.raises(EmbeddingError, match="empty"):
        make_service(fake_tokenizer, backend).embed_documents(texts)
    assert backend.passage_calls == []


def test_oversized_document_is_rejected_not_truncated(fake_tokenizer):
    backend = FakeBackend()
    service = make_service(fake_tokenizer, backend, max_tokens=12)
    assert len(service.embed_documents(["w " * 9 + "1"])) == 1  # 10 + 2 special.
    with pytest.raises(EmbeddingError, match="truncated"):
        service.embed_documents(["fine 1", "w " * 10 + "1"])
    assert len(backend.passage_calls) == 1


def test_error_messages_never_contain_document_text(fake_tokenizer):
    with pytest.raises(EmbeddingError) as caught:
        make_service(fake_tokenizer, max_tokens=4).embed_documents(["private words here 1"])
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("count", [1, 3])
def test_wrong_vector_count_rejects_the_whole_batch(fake_tokenizer, count):
    backend = FakeBackend(passages=[unit_vector(i) for i in range(count)])
    with pytest.raises(EmbeddingError, match=f"{count} vectors for 2"):
        make_service(fake_tokenizer, backend).embed_documents(["a 1", "b 2"])


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        (np.ones(383, dtype=np.float32), "shape"),
        (np.ones(385, dtype=np.float32), "shape"),
        (np.ones((1, DIMENSIONS), dtype=np.float32), "shape"),
        (np.array(1.0), "shape"),
        (np.array(["x"] * DIMENSIONS), "real-valued"),
        (np.ones(DIMENSIONS, dtype=np.complex64), "real-valued"),
        (np.zeros(DIMENSIONS, dtype=np.float32), "zero norm"),
        (np.full(DIMENSIONS, 1e300, dtype=np.float64), "non-finite"),
    ],
)
def test_malformed_vectors_are_rejected(fake_tokenizer, vector, message):
    backend = FakeBackend(passages=[vector])
    with pytest.raises(EmbeddingError, match=message):
        make_service(fake_tokenizer, backend).embed_documents(["a 1"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_one_nonfinite_value_rejects_a_later_vector(fake_tokenizer, bad):
    vector = unit_vector(1)
    vector[5] = bad
    backend = FakeBackend(passages=[unit_vector(0), vector])
    with pytest.raises(EmbeddingError, match="Vector 1 contains non-finite"):
        make_service(fake_tokenizer, backend).embed_documents(["a 1", "b 2"])


def test_generator_failure_midway_returns_no_partial_batch(fake_tokenizer):
    class FailingBackend(FakeBackend):
        def passage_embed(self, texts, **kwargs):
            yield unit_vector(0)
            raise RuntimeError("onnx failure with private text")

    with pytest.raises(EmbeddingError) as caught:
        make_service(fake_tokenizer, FailingBackend()).embed_documents(["a 1", "b 2"])
    assert "private" not in str(caught.value)


def test_query_instruction_is_applied_once(fake_tokenizer):
    backend = FakeBackend()
    service = make_service(fake_tokenizer, backend)
    service.embed_query("  How long is leave?  ")
    service.embed_query(QUERY_INSTRUCTION + "How long is leave?")
    assert backend.query_calls == [QUERY_INSTRUCTION + "How long is leave?"] * 2


@pytest.mark.parametrize("question", ["", "  \n ", QUERY_INSTRUCTION, f"  {QUERY_INSTRUCTION}  "])
def test_blank_question_is_rejected(fake_tokenizer, question):
    backend = FakeBackend()
    with pytest.raises(EmbeddingError, match="empty"):
        make_service(fake_tokenizer, backend).embed_query(question)
    assert backend.query_calls == []


def test_query_token_limit_counts_the_instruction(fake_tokenizer):
    instruction_tokens = len(fake_tokenizer.offsets(QUERY_INSTRUCTION))
    limit = instruction_tokens + 2 + 3
    backend = FakeBackend()
    service = make_service(fake_tokenizer, backend, max_tokens=limit)
    service.embed_query("one two three")
    with pytest.raises(EmbeddingError, match="shorten"):
        service.embed_query("one two three four")
    assert len(backend.query_calls) == 1


@pytest.mark.parametrize(
    ("query", "message"),
    [([], "0 vectors"), ([np.ones(385, dtype=np.float32)], "shape")],
)
def test_query_output_is_validated(fake_tokenizer, query, message):
    with pytest.raises(EmbeddingError, match=message):
        make_service(fake_tokenizer, FakeBackend(query=query)).embed_query("question")


def test_logs_counts_and_time_but_no_text(fake_tokenizer):
    stream = io.StringIO()
    configure_logging(stream=stream)
    make_service(fake_tokenizer).embed_documents(["private text 1"])
    record = json.loads(stream.getvalue())
    assert record["event"] == "embedding_completed"
    assert record["vector_count"] == 1
    assert record["duration_ms"] >= 0
    assert "private" not in stream.getvalue()


def test_cosine_similarity_is_computed_not_assumed():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 2]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [1, 0]) == pytest.approx(math.sqrt(0.5))


def test_load_fails_with_instruction_and_never_constructs_a_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A missing asset must not construct a model")

    monkeypatch.setattr(embedding_service, "TextEmbedding", forbidden)
    with pytest.raises(AssetError, match="--embedding-model"):
        load_embedding_service(Settings(model_assets_path=tmp_path / "manifest.json"))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """A manifest and local files as if preparation had already run."""
    files = {}
    for name in (*assets.EMBEDDING_TOKENIZER_FILES, assets.EMBEDDING_MODEL_FILE):
        files[name] = tmp_path / "snapshot" / name
        files[name].parent.mkdir(exist_ok=True)
        files[name].write_bytes(name.encode())
    tokenizer_hashes = {n: sha256_bytes(n.encode()) for n in assets.EMBEDDING_TOKENIZER_FILES}
    model_name = assets.EMBEDDING_MODEL_FILE
    manifest = tmp_path / "manifest.json"
    entry = {"repository": assets.EMBEDDING_EXPORT_REPOSITORY, "revision": "abc"}
    manifest.write_text(
        json.dumps(
            {
                "embedding_tokenizer": {**entry, "files": tokenizer_hashes},
                "embedding_model": {
                    **entry,
                    "files": {model_name: sha256_bytes(model_name.encode())},
                },
            }
        )
    )
    monkeypatch.setattr(assets, "_cached_file", lambda _s, _r, name: files.get(name))
    return Settings(model_assets_path=manifest), files


def test_model_resolves_to_a_verified_snapshot(prepared):
    settings, files = prepared
    asset = resolve_embedding_model(settings)
    assert asset.model_dir == files[assets.EMBEDDING_MODEL_FILE].parent


def test_tampered_weights_fail_their_checksum(prepared):
    settings, files = prepared
    files[assets.EMBEDDING_MODEL_FILE].write_bytes(b"different weights")
    with pytest.raises(AssetError, match="checksum.*--embedding-model"):
        resolve_embedding_model(settings)


def test_tampered_tokenizer_fails_model_resolution(prepared):
    settings, files = prepared
    files["tokenizer.json"].write_bytes(b"different tokenizer")
    with pytest.raises(AssetError, match="checksum"):
        resolve_embedding_model(settings)


def test_model_and_tokenizer_must_share_a_commit(prepared):
    settings, _ = prepared
    manifest = json.loads(settings.model_assets_path.read_text())
    manifest["embedding_model"]["revision"] = "other"
    settings.model_assets_path.write_text(json.dumps(manifest))
    with pytest.raises(AssetError, match="different commits"):
        resolve_embedding_model(settings)


def test_assets_id_comes_from_the_manifest_without_weights(prepared):
    settings, files = prepared
    before = embedding_assets_id(settings)
    files[assets.EMBEDDING_MODEL_FILE].unlink()
    assert embedding_assets_id(settings) == before


def test_assets_id_changes_with_the_weights_hash(prepared):
    settings, _ = prepared
    before = embedding_assets_id(settings)
    manifest = json.loads(settings.model_assets_path.read_text())
    manifest["embedding_model"]["files"][assets.EMBEDDING_MODEL_FILE] = "other"
    settings.model_assets_path.write_text(json.dumps(manifest))
    assert embedding_assets_id(settings) != before


def test_assets_id_requires_recorded_assets(tmp_path):
    with pytest.raises(AssetError, match="--embedding-model"):
        embedding_assets_id(Settings(model_assets_path=tmp_path / "manifest.json"))


def test_preparation_rejects_weights_that_differ_from_the_pin(prepared, monkeypatch):
    settings, files = prepared
    monkeypatch.setattr(assets, "prepare_embedding_tokenizer", lambda _settings: "abc")
    files[assets.EMBEDDING_MODEL_FILE].write_bytes(b"moved upstream")
    monkeypatch.setattr(
        assets,
        "hf_hub_download",
        lambda _repo, name, **kwargs: str(files[name]),
    )
    with pytest.raises(AssetError, match="manifest hash"):
        prepare_embedding_model(settings)
