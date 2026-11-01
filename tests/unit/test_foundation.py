import importlib
import io
import json
import logging
import sys

import pytest

from rag_pipeline.logging_config import configure_logging


@pytest.mark.parametrize(
    "module",
    [
        "rag_pipeline",
        "rag_pipeline.errors",
        "pymupdf",
        "fastembed",
        "onnxruntime",
        "numpy",
        "tokenizers",
        "huggingface_hub",
        "qdrant_client",
        "ollama",
        "streamlit",
        "pydantic",
        "pydantic_settings",
        "pyarrow",
    ],
)
def test_imports_without_network_or_models(module):
    assert sys.version_info[:2] == (3, 12)
    importlib.import_module(module)


def test_structured_logs_omit_sensitive_content():
    stream = io.StringIO()
    logger = configure_logging(stream=stream)
    try:
        raise ValueError("private document and secret")
    except ValueError:
        logger.exception(
            "request_failed",
            extra={
                "request_id": "request-123",
                "duration_ms": 12.5,
                "question": "private question",
                "api_key": "secret",
                "source_id": "/private/path",
            },
        )
    result = json.loads(stream.getvalue())
    assert result["event"] == "request_failed"
    assert result["error_type"] == "ValueError"
    assert result["request_id"] == "request-123"
    assert result["duration_ms"] == 12.5
    assert "source_id" not in result
    assert "private" not in stream.getvalue()
    assert "secret" not in stream.getvalue()


def test_logging_reconfiguration_does_not_duplicate_or_change_root():
    root_handlers = logging.getLogger().handlers[:]
    stream = io.StringIO()
    configure_logging(stream=stream)
    logger = configure_logging(stream=stream)
    logger.info("foundation_ready")
    assert len(stream.getvalue().splitlines()) == 1
    assert logging.getLogger().handlers == root_handlers


def test_unstructured_messages_and_nonfinite_metrics_are_omitted():
    stream = io.StringIO()
    logger = configure_logging(stream=stream)
    logger.warning("private %s", "secret", extra={"duration_ms": float("nan")})
    result = json.loads(stream.getvalue())
    assert result["event"] == "unstructured_event"
    assert "duration_ms" not in result
    assert "secret" not in stream.getvalue()
