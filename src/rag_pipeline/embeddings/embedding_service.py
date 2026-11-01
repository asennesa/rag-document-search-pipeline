"""The point where text becomes vectors.

Documents and questions use the same model. Both are checked before inference, because
FastEmbed truncates long input silently, and every output is validated before it can
reach storage. A batch either succeeds completely or raises.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding

from rag_pipeline.assets import resolve_embedding_model, resolve_embedding_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.errors import AssetError, EmbeddingError
from rag_pipeline.tokenization import QUERY_INSTRUCTION, EmbeddingTokenizer, TextTokenizer

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    def passage_embed(self, texts: Iterable[str], **kwargs: object) -> Iterable[np.ndarray]: ...

    def query_embed(self, query: str, **kwargs: object) -> Iterable[np.ndarray]: ...


class EmbeddingService:
    """Holds one backend, so the model is loaded once however often it is called."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        tokenizer: TextTokenizer,
        *,
        model_name: str,
        dimensions: int,
        max_tokens: int,
        batch_size: int,
        load_ms: float = 0.0,
    ) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self.model_name = model_name
        self.dimensions = dimensions
        self.load_ms = load_ms  # Cold-start cost, kept apart from inference timings.
        self._max_tokens = max_tokens
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages in order. Empty input gives an empty result without inference."""
        if isinstance(texts, str):
            raise EmbeddingError("embed_documents expects a sequence of texts, not one string")
        if not texts:
            return []
        for index, text in enumerate(texts):
            if not text.strip():
                raise EmbeddingError(f"Text at position {index} is empty")
            if self._input_tokens(text) > self._max_tokens:
                raise EmbeddingError(
                    f"Text at position {index} exceeds {self._max_tokens} tokens and would "
                    "be truncated; chunk it first"
                )
        return self._run(
            lambda: self._backend.passage_embed(list(texts), batch_size=self._batch_size),
            expected=len(texts),
        )

    def embed_query(self, question: str) -> list[float]:
        """Embed a question with the BGE retrieval instruction applied exactly once."""
        text = question.strip().removeprefix(QUERY_INSTRUCTION.rstrip()).strip()
        if not text:
            raise EmbeddingError("Question is empty")
        prefixed = QUERY_INSTRUCTION + text
        if self._input_tokens(prefixed) > self._max_tokens:
            raise EmbeddingError(f"Question exceeds {self._max_tokens} tokens; please shorten it")
        return self._run(lambda: self._backend.query_embed(prefixed), expected=1)[0]

    def _input_tokens(self, text: str) -> int:
        return len(self._tokenizer.offsets(text)) + self._tokenizer.special_token_count

    def _run(self, infer, expected: int) -> list[list[float]]:
        started = time.perf_counter()
        try:
            # FastEmbed returns a generator: inference happens while it is consumed.
            raw = list(infer())
        except Exception as error:  # ONNX Runtime raises its own unrelated types.
            raise EmbeddingError("Embedding inference failed") from error
        vectors = self._validated(raw, expected)
        logger.info(
            "embedding_completed",
            extra={
                "vector_count": len(vectors),
                "duration_ms": (time.perf_counter() - started) * 1000,
            },
        )
        return vectors

    def _validated(self, raw: list[np.ndarray], expected: int) -> list[list[float]]:
        if len(raw) != expected:
            raise EmbeddingError(f"Model returned {len(raw)} vectors for {expected} inputs")
        vectors = []
        for index, item in enumerate(raw):
            vector = np.asarray(item)
            if not np.issubdtype(vector.dtype, np.number) or np.iscomplexobj(vector):
                raise EmbeddingError(f"Vector {index} is not real-valued")
            if vector.shape != (self.dimensions,):
                raise EmbeddingError(
                    f"Vector {index} has shape {vector.shape}, expected ({self.dimensions},)"
                )
            with np.errstate(over="ignore"):  # Overflow becomes inf, rejected just below.
                vector = vector.astype(np.float32)
            if not np.isfinite(vector).all():
                raise EmbeddingError(f"Vector {index} contains non-finite values")
            if not np.any(vector):
                raise EmbeddingError(f"Vector {index} has zero norm")
            vectors.append(vector.tolist())
        return vectors


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_embedding_service(settings: Settings) -> EmbeddingService:
    """Load the verified local snapshot. FastEmbed gets a path, so it cannot download."""
    model_asset = resolve_embedding_model(settings)
    tokenizer_asset = resolve_embedding_tokenizer(settings)
    started = time.perf_counter()
    try:
        backend = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=str(settings.embedding_cache_dir),
            threads=settings.embedding_threads,
            providers=["CPUExecutionProvider"],
            specific_model_path=str(model_asset.model_dir),
        )
    except Exception as error:  # ONNX Runtime raises its own unrelated types.
        raise AssetError("Embedding model failed to load; prepare it again") from error
    return EmbeddingService(
        backend,
        EmbeddingTokenizer(tokenizer_asset.tokenizer_file),
        model_name=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        max_tokens=settings.embedding_max_tokens,
        batch_size=settings.embedding_batch_size,
        load_ms=(time.perf_counter() - started) * 1000,
    )
