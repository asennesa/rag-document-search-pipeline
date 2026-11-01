"""Adapter for local Ollama chat. It knows nothing about retrieval.

The request exposes no tools, and models are never pulled automatically. A cloud-backed
model is a server setting, so OLLAMA_NO_CLOUD=1 must be set on the Ollama server.
"""

import hashlib
import time
from dataclasses import dataclass

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from rag_pipeline.errors import GenerationError
from rag_pipeline.rag.generation_tokenizer import SUPPORTED_TEMPLATE_SHA256


class GeneratedAnswer(BaseModel):
    """The schema makes the model state its sources. A prompt rule alone was ignored in
    most real answers. The IDs are still the model's claim and are validated afterwards."""

    answer: str
    citations: list[str]
    insufficient_evidence: bool


@dataclass(frozen=True)
class LLMResult:
    answer: str
    insufficient_evidence: bool
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    citations: tuple[str, ...] = ()


class LLMService:
    def __init__(
        self,
        client: ollama.Client,
        *,
        model: str,
        num_ctx: int,
        max_output_tokens: int,
        temperature: float,
        template_sha256: str = SUPPORTED_TEMPLATE_SHA256,
    ) -> None:
        self._client = client
        self._model = model
        self._num_ctx = num_ctx
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._template_sha256 = template_sha256
        self._verified = False

    def verify(self) -> None:
        """Check the model exists and that token budgeting is valid for its template."""
        if self._verified:
            return
        details = self._request(lambda: self._client.show(self._model))
        template = (details.template or "").encode()
        if hashlib.sha256(template).hexdigest() != self._template_sha256:
            raise GenerationError(
                "model_unverified",
                "Model template differs from the one the token budget was verified against",
            )
        limits = [
            value
            for key, value in (details.modelinfo or {}).items()
            if key.endswith(".context_length") and isinstance(value, int)
        ]
        if limits and self._num_ctx > min(limits):
            raise GenerationError(
                "model_unverified", "OLLAMA_NUM_CTX exceeds the model's context length"
            )
        self._verified = True

    def generate(self, system: str, user: str) -> LLMResult:
        self.verify()
        started = time.perf_counter()
        response = self._request(
            lambda: self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=False,
                think=False,  # This instruct model has no reasoning channel to reveal.
                format=GeneratedAnswer.model_json_schema(),
                options={
                    "temperature": self._temperature,
                    "num_ctx": self._num_ctx,
                    "num_predict": self._max_output_tokens,
                },
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000
        # Checked before parsing: cut-off JSON is an incomplete answer, not a malformed one.
        if response.done_reason == "length":
            raise GenerationError(
                "incomplete", "The model reached the output limit before finishing"
            )
        try:
            parsed = GeneratedAnswer.model_validate_json(response.message.content or "")
        except ValidationError as error:
            raise GenerationError(
                "malformed_response", "The model did not return the expected JSON"
            ) from error
        if not parsed.answer.strip():
            raise GenerationError("malformed_response", "The model returned an empty answer")
        return LLMResult(
            parsed.answer.strip(),
            parsed.insufficient_evidence,
            response.prompt_eval_count,
            response.eval_count,
            latency_ms,
            tuple(parsed.citations),
        )

    def _request(self, call):
        try:
            return call()
        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise GenerationError(
                    "model_missing", f"Model is not installed. Run: ollama pull {self._model}"
                ) from error
            raise GenerationError("ollama_unavailable", "Ollama rejected the request") from error
        except httpx.TimeoutException as error:
            raise GenerationError(
                "timeout", "Ollama did not answer in time; raise OLLAMA_TIMEOUT_SECONDS"
            ) from error
        except (ConnectionError, httpx.HTTPError) as error:
            raise GenerationError(
                "ollama_unavailable", "Ollama is unreachable; is it running? Try: ollama serve"
            ) from error
