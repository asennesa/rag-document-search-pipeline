import pytest

from rag_pipeline.errors import EvaluationError
from rag_pipeline.evaluation.dataset import EvalQuestion
from rag_pipeline.evaluation.rag_metrics import (
    RagRow,
    citation_summary,
    classify_failure,
    correctness_summary,
    evidence_summary,
    refusal_summary,
    rows_from_json,
    rows_to_json,
    summarize,
    validate_judgments,
)


def question(id="q", category="straightforward", relevant=("a.md",), required=()):
    answerable = category != "unanswerable"
    return EvalQuestion(
        id=id,
        category=category,
        question="Question?",
        reference_answer="Answer" if answerable else None,
        relevant_sources=relevant if answerable else (),
        required_sources=required,
        supporting_passages=("x",) if answerable else (),
    )


def row(id="q", status="answered", cited=("S1",), unknown=(), malformed=(), retrieved=("a.md",)):
    return RagRow(
        id=id,
        category="straightforward",
        question="Question?",
        reference_answer="Answer",
        status=status,
        answer="Answer" if status != "ollama_unavailable" else None,
        cited_ids=cited,
        unknown_ids=unknown,
        malformed=malformed,
        retrieved_sources=retrieved,
        context_sources=retrieved,
        cited_sources=retrieved,
        retrieval_ms=5.0,
        generation_ms=1000.0,
        prompt_tokens=100,
        output_tokens=20,
    )


def test_citation_validity_counts_references():
    rows = [row("1", cited=("S1", "S2")), row("2", "citation_invalid", ("S9",), ("S9",))]
    result = citation_summary(rows)
    assert (result["emitted_references"], result["valid_references"]) == (3, 2)
    assert result["validity"] == pytest.approx(2 / 3)


def test_malformed_references_count_against_validity():
    result = citation_summary([row(status="citation_invalid", cited=(), malformed=("[S]",))])
    assert (result["emitted_references"], result["validity"]) == (1, 0.0)
    assert result["uncited_substantive_answers"] == 0


def test_zero_citations_are_not_vacuously_valid():
    result = citation_summary([row(status="citation_invalid", cited=())])
    assert result["validity"] is None
    assert result["uncited_substantive_answers"] == 1


def test_refusals_do_not_count_as_uncited_answers():
    result = citation_summary([row(status="insufficient_evidence", cited=())])
    assert result["substantive_answers"] == 0


def test_refusal_rates_use_separate_denominators():
    by_id = {
        "a": question("a"),
        "b": question("b"),
        "u1": question("u1", "unanswerable"),
        "u2": question("u2", "unanswerable"),
    }
    rows = [
        row("a"),
        row("b", "insufficient_evidence", ()),
        row("u1", "insufficient_evidence", ()),
        row("u2"),
    ]
    result = refusal_summary(rows, by_id)
    assert result["correct_refusal_rate"] == 0.5
    assert result["incorrect_refusal_rate"] == 0.5


def test_no_context_counts_as_a_refusal():
    by_id = {"u": question("u", "unanswerable")}
    assert refusal_summary([row("u", "no_context", ())], by_id)["correct_refusals"] == 1


def test_empty_refusal_denominators_are_none():
    assert refusal_summary([], {})["correct_refusal_rate"] is None


def multi():
    return question("m", "multi_passage", ("a.md", "b.md"), ("a.md", "b.md"))


def test_evidence_needs_every_required_document():
    by_id = {"m": multi()}
    partial = evidence_summary([row("m", retrieved=("a.md",))], by_id)
    full = evidence_summary([row("m", retrieved=("a.md", "b.md"))], by_id)
    assert partial["all_retrieved"] == 0 and full["all_retrieved"] == 1


def test_stage_retrieval_when_no_relevant_document_was_found():
    assert classify_failure(row(retrieved=("x.md",)), question(), 2) == "retrieval"


def test_stage_retrieval_when_one_required_document_is_missing():
    assert classify_failure(row("m", retrieved=("a.md",)), multi(), 2) == "retrieval"


def test_stage_context_omission_when_retrieved_but_not_sent():
    item = row()
    omitted = RagRow(**{**item.__dict__, "context_sources": ()})
    assert classify_failure(omitted, question(), 0) == "context_omission"


def test_stage_generation_when_evidence_was_available():
    assert classify_failure(row(), question(), 1) == "generation"
    refused = row(status="insufficient_evidence", cited=())
    assert classify_failure(refused, question(), None) == "generation"


def test_answering_an_unanswerable_question_is_a_generation_failure():
    assert classify_failure(row(), question("u", "unanswerable"), None) == "generation"
    refused = row(status="insufficient_evidence", cited=())
    assert classify_failure(refused, question("u", "unanswerable"), None) is None


def test_full_score_has_no_failure_and_missing_score_is_unjudged():
    assert classify_failure(row(), question(), 2) is None
    assert classify_failure(row(), question(), None) == "unjudged"


def test_correctness_without_judgments_is_not_available():
    assert correctness_summary([row()], None)["status"].startswith("N/A")


def test_correctness_hand_computed_and_partial_review_is_visible():
    rows = [row("a"), row("b"), row("c")]
    result = correctness_summary(rows, {"a": 2, "b": 0})
    assert result["mean_score"] == 1.0
    assert (result["judged"], result["total"]) == (2, 3)


def judgments(**fields):
    base = {"reviewer": "A", "reviewer_kind": "human", "judgments": {"q": {"score": 2}}}
    return {**base, **fields}


def test_valid_judgments_parse():
    assert validate_judgments(judgments(), [row()]) == {"q": 2}


def test_blank_scores_stay_unjudged():
    data = judgments(judgments={"q": {"score": None}})
    assert validate_judgments(data, [row()]) == {}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"reviewer": ""}, "reviewer"),
        ({"reviewer_kind": "model"}, "reviewer"),
        ({"judgments": []}, "mapping"),
        ({"judgments": {"other": {"score": 1}}}, "unknown"),
        ({"judgments": {"q": {"score": 3}}}, "0, 1, or 2"),
        ({"judgments": {"q": {"score": True}}}, "0, 1, or 2"),
        ({"judgments": {"q": {"score": 1.5}}}, "0, 1, or 2"),
    ],
)
def test_invalid_judgments_are_rejected(change, message):
    with pytest.raises(EvaluationError, match=message):
        validate_judgments(judgments(**change), [row()])


def test_summary_excludes_failed_generations_from_latency():
    questions = [question("a"), question("b")]
    rows = [row("a"), row("b", "ollama_unavailable", ())]
    result = summarize(rows, questions)
    assert result["generation_latency"]["count"] == 1
    assert result["retrieval_latency"]["count"] == 2
    assert result["status_counts"] == {"answered": 1, "ollama_unavailable": 1}


def test_rows_survive_a_json_round_trip():
    rows = [row()]
    assert rows_from_json(rows_to_json(rows)) == rows
