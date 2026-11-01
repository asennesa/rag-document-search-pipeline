"""RAG metrics. Deterministic checks are computed; correctness comes from reviewer scores.

Automated string comparison is not a correctness measure, so without judgments correctness
is reported as N/A. A judged score uses this rubric:

- 0: unsupported or wrong.
- 1: partially supported, or missing part of the reference answer.
- 2: fully supported by the sources and matching the reference answer.
"""

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from rag_pipeline.errors import EvaluationError
from rag_pipeline.evaluation.dataset import EvalQuestion
from rag_pipeline.evaluation.retrieval_metrics import latency_summary


@dataclass(frozen=True)
class RagRow:
    id: str
    category: str
    question: str
    reference_answer: str | None
    status: str
    answer: str | None
    cited_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    malformed: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    context_sources: tuple[str, ...]
    cited_sources: tuple[str, ...]
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int | None
    output_tokens: int | None


def build_row(question: EvalQuestion, response) -> RagRow:
    citation = response.citation
    cited = citation.cited_ids if citation else ()
    return RagRow(
        id=question.id,
        category=question.category,
        question=question.question,
        reference_answer=question.reference_answer,
        status=response.status,
        answer=response.answer,
        cited_ids=cited,
        unknown_ids=citation.unknown_ids if citation else (),
        malformed=citation.malformed if citation else (),
        retrieved_sources=tuple(dict.fromkeys(p.source for p in response.retrieved)),
        context_sources=tuple(dict.fromkeys(s.passage.source for s in response.sources)),
        cited_sources=tuple(
            dict.fromkeys(s.passage.source for s in response.sources if s.source_id in cited)
        ),
        retrieval_ms=response.retrieval_time_ms,
        generation_ms=response.generation_time_ms,
        prompt_tokens=response.prompt_tokens,
        output_tokens=response.output_tokens,
    )


def citation_summary(rows: Sequence[RagRow]) -> dict:
    """Validity is over emitted references, so zero citations cannot look like 100%."""
    emitted = sum(len(r.cited_ids) + len(r.malformed) for r in rows)
    invalid = sum(len(r.unknown_ids) + len(r.malformed) for r in rows)
    substantive = [r for r in rows if r.status in ("answered", "citation_invalid")]
    uncited = sum(1 for r in substantive if not r.cited_ids and not r.malformed)
    return {
        "emitted_references": emitted,
        "valid_references": emitted - invalid,
        "validity": (emitted - invalid) / emitted if emitted else None,
        "substantive_answers": len(substantive),
        "uncited_substantive_answers": uncited,
    }


def refusal_summary(rows: Sequence[RagRow], by_id: Mapping[str, EvalQuestion]) -> dict:
    unanswerable = [r for r in rows if not by_id[r.id].answerable]
    answerable = [r for r in rows if by_id[r.id].answerable]
    refused = ("insufficient_evidence", "no_context")
    correct = sum(1 for r in unanswerable if r.status in refused)
    incorrect = sum(1 for r in answerable if r.status in refused)
    return {
        "unanswerable_questions": len(unanswerable),
        "correct_refusals": correct,
        "correct_refusal_rate": correct / len(unanswerable) if unanswerable else None,
        "answerable_questions": len(answerable),
        "incorrect_refusals": incorrect,
        "incorrect_refusal_rate": incorrect / len(answerable) if answerable else None,
    }


def evidence_summary(rows: Sequence[RagRow], by_id: Mapping[str, EvalQuestion]) -> dict:
    """For multi-passage questions, was every required document retrieved and retained?"""
    multi = [r for r in rows if by_id[r.id].required_sources]
    retrieved = sum(
        1 for r in multi if set(by_id[r.id].required_sources) <= set(r.retrieved_sources)
    )
    retained = sum(1 for r in multi if set(by_id[r.id].required_sources) <= set(r.context_sources))
    return {"questions": len(multi), "all_retrieved": retrieved, "all_retained": retained}


def classify_failure(row: RagRow, question: EvalQuestion, score: int | None) -> str | None:
    """Assign one earliest stage, so a retrieval miss is not blamed on the model."""
    refused = row.status in ("insufficient_evidence", "no_context")
    if not question.answerable:
        return None if refused else "generation"
    needed = set(question.required_sources)
    if needed:
        if not needed <= set(row.retrieved_sources):
            return "retrieval"
        if not needed <= set(row.context_sources):
            return "context_omission"
    else:
        relevant = set(question.relevant_sources)
        if not relevant & set(row.retrieved_sources):
            return "retrieval"
        if not relevant & set(row.context_sources):
            return "context_omission"
    if row.status != "answered" or (score is not None and score < 2):
        return "generation"
    return "unjudged" if score is None else None


def validate_judgments(data: Mapping, rows: Sequence[RagRow]) -> dict[str, int]:
    if data.get("reviewer_kind") not in ("human", "agent") or not data.get("reviewer"):
        raise EvaluationError("Judgments must name a reviewer and reviewer_kind")
    judgments = data.get("judgments")
    if not isinstance(judgments, Mapping):
        raise EvaluationError("Judgments must contain a judgments mapping")
    known = {row.id for row in rows}
    scores: dict[str, int] = {}
    for question_id, entry in judgments.items():
        if question_id not in known:
            raise EvaluationError(f"Judgment names an unknown question: {question_id}")
        score = entry.get("score") if isinstance(entry, Mapping) else None
        if score is None:
            continue  # Left blank by the reviewer, which stays N/A.
        if type(score) is not int or score not in (0, 1, 2):
            raise EvaluationError(f"Score for {question_id} must be 0, 1, or 2")
        scores[question_id] = score
    return scores


def correctness_summary(rows: Sequence[RagRow], scores: Mapping[str, int] | None) -> dict:
    if not scores:
        return {"status": "N/A: no reviewed judgments", "judged": 0, "total": len(rows)}
    judged = [scores[row.id] for row in rows if row.id in scores]
    counts = Counter(judged)
    return {
        "judged": len(judged),
        "total": len(rows),
        "mean_score": statistics.fmean(judged) if judged else None,
        "fully_supported": counts[2],
        "partially_supported": counts[1],
        "unsupported_or_wrong": counts[0],
    }


def summarize(
    rows: Sequence[RagRow],
    questions: Sequence[EvalQuestion],
    scores: Mapping[str, int] | None = None,
) -> dict:
    by_id = {question.id: question for question in questions}
    failures = Counter(
        classify_failure(row, by_id[row.id], (scores or {}).get(row.id)) for row in rows
    )
    failures.pop(None, None)
    return {
        "questions": len(rows),
        "status_counts": dict(Counter(row.status for row in rows)),
        "citations": citation_summary(rows),
        "refusals": refusal_summary(rows, by_id),
        "evidence": evidence_summary(rows, by_id),
        "correctness": correctness_summary(rows, scores),
        "failure_stages": dict(failures),
        "retrieval_latency": latency_summary([row.retrieval_ms for row in rows]),
        "generation_latency": latency_summary(
            [row.generation_ms for row in rows if row.answer is not None]
        ),
    }


def evaluate_rag(rag, questions: Sequence[EvalQuestion], top_k: int) -> list[RagRow]:
    return [build_row(question, rag.ask(question.question, top_k)) for question in questions]


def rows_to_json(rows: Sequence[RagRow]) -> list[dict]:
    return [asdict(row) for row in rows]


def rows_from_json(items: Sequence[Mapping]) -> list[RagRow]:
    fields = RagRow.__dataclass_fields__
    tuples = [name for name, field in fields.items() if "tuple" in str(field.type)]
    return [RagRow(**{**item, **{name: tuple(item[name]) for name in tuples}}) for item in items]
