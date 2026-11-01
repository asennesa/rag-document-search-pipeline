import pytest

from rag_pipeline.cli import run_cli
from rag_pipeline.config import Settings
from rag_pipeline.errors import VectorStoreError


def test_successful_exit_code_passes_through():
    assert run_cli(lambda: 1) == 1


def test_pipeline_error_is_a_clean_message(capsys):
    def fail():
        raise VectorStoreError("Qdrant is unreachable")

    assert run_cli(fail) == 2
    assert capsys.readouterr().err == "error: Qdrant is unreachable\n"


def test_invalid_configuration_names_the_setting_but_not_the_value(capsys, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "secret-looking-value")
    assert run_cli(lambda: Settings() and 0) == 2
    message = capsys.readouterr().err
    assert "chunk_size" in message
    assert "secret-looking-value" not in message


def test_unexpected_errors_are_not_hidden():
    def fail():
        raise KeyError("a real bug")

    with pytest.raises(KeyError):
        run_cli(fail)
