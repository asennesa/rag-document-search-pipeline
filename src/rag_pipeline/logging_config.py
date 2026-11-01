"""Structured application logging with an allowlist of diagnostic fields.

Use static snake_case event names and opaque IDs, never document text or paths.
Exception messages and tracebacks are deliberately omitted from normal logs.
"""

import json
import logging
import math
import re
from datetime import UTC, datetime
from typing import TextIO

_IDENTIFIER = re.compile(r"[a-zA-Z0-9_-]{1,80}\Z")
_ID_FIELDS = ("run_id", "request_id", "stage", "source_id", "document_id", "error_category")
_NUMBER_FIELDS = (
    "duration_ms",
    "embedding_ms",
    "retrieval_ms",
    "context_ms",
    "generation_ms",
    "document_count",
    "chunk_count",
    "vector_count",
    "deleted_count",
    "failure_count",
)


class JsonFormatter(logging.Formatter):
    """Emit known metadata only; arbitrary messages become unstructured_event."""

    def format(self, record: logging.LogRecord) -> str:
        event = record.msg
        if (
            not isinstance(event, str)
            or record.args
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", event)
        ):
            event = "unstructured_event"
        output = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": event,
        }
        for key in _ID_FIELDS:
            value = getattr(record, key, None)
            if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
                output[key] = value
        for key in _NUMBER_FIELDS:
            value = getattr(record, key, None)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                output[key] = value
        if record.exc_info and record.exc_info[0]:
            output["error_type"] = record.exc_info[0].__name__
        return json.dumps(output, allow_nan=False)


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> logging.Logger:
    """Configure only the application logger; repeated calls replace its handlers."""
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
