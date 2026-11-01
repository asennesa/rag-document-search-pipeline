"""Shared error boundary for pipeline components."""


class PipelineError(Exception):
    """An expected pipeline failure with an actionable, content-free message."""


class DocumentError(PipelineError):
    """A document that cannot be processed; `category` is a stable reporting label."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class ChunkingError(DocumentError):
    """Text that cannot be split within the token limits without losing content."""

    def __init__(self, message: str) -> None:
        super().__init__("chunking_failed", message)


class AssetError(PipelineError):
    """A model asset that is missing, unpinned, or does not match its manifest."""


class EmbeddingError(PipelineError):
    """Input that cannot be embedded safely, or model output that fails validation."""


class VectorStoreError(PipelineError):
    """Qdrant is unavailable, rejected a request, or holds incompatible data."""


class WriterLockError(PipelineError):
    """Another local process is already writing to this collection."""


class SearchError(PipelineError):
    """A search request that fails validation."""


class GenerationError(PipelineError):
    """A generation failure; `category` becomes the response status."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class EvaluationError(PipelineError):
    """Evaluation labels or inputs that cannot produce a trustworthy result."""
