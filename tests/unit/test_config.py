from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_pipeline.config import Settings


def test_defaults_and_example_are_consistent():
    example = Path(__file__).resolve().parents[2] / ".env.example"
    assert Settings(_env_file=example) == Settings()


@pytest.mark.parametrize(
    "values",
    [
        {"chunk_size": 511},
        {"chunk_overlap": -1},
        {"chunk_overlap": 256},
        {"search_top_k": 21},
        {"embedding_batch_size": 0},
        {"embedding_model": "unknown"},
        {"embedding_dimensions": 768},
        {"qdrant_timeout_seconds": float("inf")},
        {"rag_temperature": float("nan")},
        {"ollama_num_ctx": 640},
        {"qdrant_hnsw_ef_construct": 8},
        {"qdrant_url": "ftp://localhost"},
        {"qdrant_url": "http://user:secret@localhost"},
        {"ollama_base_url": "http://localhost/api"},
        {"qdrant_url": "http://localhost/?key=x"},
        {"documents_dir": ""},
        {"pdf_max_pages": 0},
        {"upload_max_bytes": -1},
        {"log_level": "TRACE"},
        {"qdrant_collection": ""},
        {"source_namespace": "../private"},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        Settings(**values)


def test_precedence(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("SEARCH_TOP_K=3\nLOG_LEVEL=debug\n", encoding="utf-8")
    assert Settings().search_top_k == 3
    assert Settings().log_level == "DEBUG"
    monkeypatch.setenv("SEARCH_TOP_K", "7")
    assert Settings().search_top_k == 7
    assert Settings(search_top_k=9).search_top_k == 9


def test_secret_representation():
    settings = Settings(qdrant_api_key="private-token")
    assert "private-token" not in repr(settings)
    assert "private-token" not in settings.model_dump_json()
    assert settings.qdrant_api_key.get_secret_value() == "private-token"


def test_unknown_dotenv_option_is_actionable(tmp_path):
    (tmp_path / ".env").write_text("SEARCH_TOPK=5\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="search_topk"):
        Settings()


def test_configuration_does_not_create_directories(tmp_path):
    Settings(documents_dir=tmp_path / "missing")
    assert list(tmp_path.iterdir()) == []


def test_validation_message_hides_secret_input():
    with pytest.raises(ValidationError) as error:
        Settings(qdrant_url="http://user:private-token@localhost")
    assert "private-token" not in str(error.value)
