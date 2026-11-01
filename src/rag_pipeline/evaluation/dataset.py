"""Labeled questions, checked against the corpus before anything runs.

Labels describe specific document text, so a corpus edit can silently invalidate them.
A recorded corpus hash and quote checks turn that into a clear failure.
"""

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from rag_pipeline.errors import EvaluationError
from rag_pipeline.identifiers import make_processing_fingerprint
from rag_pipeline.models import Document

Category = Literal["straightforward", "paraphrased", "multi_passage", "unanswerable"]
MINIMUM_COUNTS = {
    "straightforward": 10,
    "paraphrased": 5,
    "multi_passage": 5,
    "unanswerable": 5,
}


class EvalQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    category: Category
    question: str
    reference_answer: str | None
    relevant_sources: tuple[str, ...]  # Exhaustive: every document that answers it.
    required_sources: tuple[str, ...]  # All needed together; empty unless multi-passage.
    supporting_passages: tuple[str, ...]

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.category == "unanswerable":
            if self.reference_answer or self.relevant_sources or self.supporting_passages:
                raise ValueError("Unanswerable questions cannot have answers or sources")
            return self
        if not (self.reference_answer and self.relevant_sources and self.supporting_passages):
            raise ValueError("Answerable questions need an answer, sources, and passages")
        if not set(self.required_sources) <= set(self.relevant_sources):
            raise ValueError("Required sources must be relevant sources")
        if (self.category == "multi_passage") != (len(self.required_sources) >= 2):
            raise ValueError("Exactly multi-passage questions need two or more required sources")
        return self

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_questions(path: Path) -> list[EvalQuestion]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        questions = [EvalQuestion(**json.loads(line)) for line in lines if line.strip()]
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise EvaluationError(f"Evaluation labels are unreadable or invalid: {path}") from error
    ids = Counter(question.id for question in questions)
    if len(ids) != len(questions):
        raise EvaluationError("Evaluation question IDs must be unique")
    counts = Counter(question.category for question in questions)
    for category, minimum in MINIMUM_COUNTS.items():
        if counts[category] < minimum:
            raise EvaluationError(f"Need at least {minimum} {category} questions")
    return questions


def corpus_fingerprint(documents: Iterable[Document]) -> str:
    return make_processing_fingerprint(
        documents=sorted((document.source, document.document_hash) for document in documents)
    )


def _normalized(text: str) -> str:
    return " ".join(text.split())


def validate_labels(
    questions: Iterable[EvalQuestion],
    documents: Iterable[Document],
    expected_corpus: str,
) -> None:
    """Fail on any stale reference, since metrics from stale labels look fine but mean nothing."""
    documents = list(documents)
    if corpus_fingerprint(documents) != expected_corpus:
        raise EvaluationError(
            "The sample documents changed since these labels were written. Review "
            "data/evaluation/questions.jsonl, then update data/evaluation/corpus_lock.json"
        )
    text = {
        document.source: _normalized(" ".join(page.cleaned_text for page in document.pages))
        for document in documents
    }
    for question in questions:
        for source in question.relevant_sources:
            if source not in text:
                raise EvaluationError(f"{question.id} names an unknown source: {source}")
        relevant_text = [text[source] for source in question.relevant_sources]
        for passage in question.supporting_passages:
            if not any(_normalized(passage) in document for document in relevant_text):
                raise EvaluationError(f"{question.id} has a passage not found in its sources")


def read_corpus_lock(path: Path) -> str:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["corpus_fingerprint"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise EvaluationError(f"Corpus lock is unreadable: {path}") from error
