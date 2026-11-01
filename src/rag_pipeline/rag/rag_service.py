"""Question to grounded answer. Reuses retrieval unchanged.

Generation problems become statuses so retrieved passages are still returned. Retrieval
failures raise, since there is nothing useful to show without them.
"""

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import ollama

from rag_pipeline.assets import resolve_generation_tokenizer
from rag_pipeline.config import Settings
from rag_pipeline.errors import GenerationError
from rag_pipeline.models import BuiltContext, RagResponse, RagStatus, SearchResponse
from rag_pipeline.rag.citation_validator import validate_citations
from rag_pipeline.rag.context_builder import ContextBuilder
from rag_pipeline.rag.generation_tokenizer import GenerationTokenizer
from rag_pipeline.rag.llm_service import LLMResult, LLMService
from rag_pipeline.retrieval.semantic_search import load_search_service

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def search(
        self, question: str, top_k: int | None = None, source: str | None = None
    ) -> SearchResponse: ...


class ContextAssembler(Protocol):
    def build(self, question, passages) -> BuiltContext: ...


class Generator(Protocol):
    def generate(self, system: str, user: str) -> LLMResult: ...


@dataclass(frozen=True)
class _Outcome:
    status: RagStatus
    answer: str | None = None
    message: str | None = None
    citation: object = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class RAGService:
    def __init__(
        self, retriever: Retriever, builder: ContextAssembler, llm: Generator, *, model: str
    ) -> None:
        self._retriever = retriever
        self._builder = builder
        self._llm = llm
        self._model = model

    def ask(
        self, question: str, top_k: int | None = None, source: str | None = None
    ) -> RagResponse:
        started = time.perf_counter()
        search = self._retriever.search(question, top_k, source)

        context_started = time.perf_counter()
        context: BuiltContext | None = None
        context_error: GenerationError | None = None
        try:
            context = self._builder.build(search.query, search.results)
        except GenerationError as error:
            context_error = error
        context_ms = (time.perf_counter() - context_started) * 1000

        generation_started = time.perf_counter()
        outcome = self._decide(context, context_error)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        response = RagResponse(
            question=search.query,
            status=outcome.status,
            answer=outcome.answer,
            message=outcome.message,
            citation=outcome.citation,
            sources=context.sources if context else (),
            retrieved=search.results,
            omitted=context.omitted if context else (),
            model=self._model,
            prompt_tokens=outcome.prompt_tokens,
            output_tokens=outcome.output_tokens,
            retrieval_time_ms=search.retrieval_time_ms,
            context_time_ms=context_ms,
            generation_time_ms=generation_ms,
            total_time_ms=(time.perf_counter() - started) * 1000,
        )
        logger.info(
            "rag_completed",
            extra={
                "error_category": response.status,
                "chunk_count": len(response.sources),
                "retrieval_ms": response.retrieval_time_ms,
                "context_ms": response.context_time_ms,
                "generation_ms": response.generation_time_ms,
                "duration_ms": response.total_time_ms,
            },
        )
        return response

    def _decide(
        self, context: BuiltContext | None, context_error: GenerationError | None
    ) -> _Outcome:
        if context_error is not None:
            return _Outcome(context_error.category, message=str(context_error))
        if not context.sources:
            # Nothing to ground an answer on, so the model is not called at all.
            return _Outcome("no_context", message="No usable passages were retrieved")
        try:
            result = self._llm.generate(context.system_prompt, context.user_prompt)
        except GenerationError as error:
            return _Outcome(error.category, message=str(error))

        citation = validate_citations(
            result.answer,
            {source.source_id for source in context.sources},
            insufficient_evidence=result.insufficient_evidence,
            listed_ids=result.citations,
        )
        if citation.status in ("invalid", "missing"):
            status: RagStatus = "citation_invalid"
            message = "The answer's citations failed validation; do not treat it as grounded"
        elif result.insufficient_evidence:
            status, message = "insufficient_evidence", None
        else:
            status, message = "answered", None
        return _Outcome(
            status, result.answer, message, citation, result.prompt_tokens, result.output_tokens
        )


def load_rag_service(settings: Settings) -> RAGService:
    """Uses the search factory, so RAG shares its collection checks. Contacts no LLM yet."""
    retriever = load_search_service(settings)
    tokenizer = GenerationTokenizer(resolve_generation_tokenizer(settings))
    builder = ContextBuilder(
        tokenizer,
        num_ctx=settings.ollama_num_ctx,
        max_context_tokens=settings.rag_max_context_tokens,
        max_output_tokens=settings.rag_max_output_tokens,
        safety_tokens=settings.rag_context_safety_tokens,
    )
    client = ollama.Client(
        host=str(settings.ollama_base_url), timeout=settings.ollama_timeout_seconds
    )
    llm = LLMService(
        client,
        model=settings.ollama_model,
        num_ctx=settings.ollama_num_ctx,
        max_output_tokens=settings.rag_max_output_tokens,
        temperature=settings.rag_temperature,
    )
    return RAGService(retriever, builder, llm, model=settings.ollama_model)
