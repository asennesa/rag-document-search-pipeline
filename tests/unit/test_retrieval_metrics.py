import json

import pytest

from rag_pipeline.errors import EvaluationError
from rag_pipeline.evaluation.dataset import (
    EvalQuestion,
    corpus_fingerprint,
    load_questions,
    read_corpus_lock,
    validate_labels,
)
from rag_pipeline.evaluation.retrieval_metrics import (
    aggregate,
    evaluate_retrieval,
    hit_at_k,
    latency_summary,
    ranked_documents,
    recall_at_k,
    reciprocal_rank_at_k,
    score_question,
)
from rag_pipeline.models import Document, Page, RetrievedPassage, SearchResponse


def question(id="q", category="straightforward", relevant=("a.md",), required=()):
    answerable = category != "unanswerable"
    return EvalQuestion(
        id=id,
        category=category,
        question="Question?",
        reference_answer="Answer" if answerable else None,
        relevant_sources=relevant if answerable else (),
        required_sources=required,
        supporting_passages=("quoted text",) if answerable else (),
    )


def test_duplicates_collapse_to_the_best_rank():
    assert ranked_documents(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_hit_counts_any_relevant_document_within_k():
    assert hit_at_k(["x", "a"], ["a"], 2) == 1.0
    assert hit_at_k(["x", "a"], ["a"], 1) == 0.0


def test_recall_hand_computed():
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c", "d"], 3) == 0.5
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c", "d"], 1) == 0.25


def test_recall_refuses_a_zero_denominator():
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(["a"], [], 5)


@pytest.mark.parametrize(
    ("ranked", "k", "expected"),
    [(["a"], 5, 1.0), (["x", "a"], 5, 0.5), (["x", "y", "z", "w", "a"], 5, 0.2)],
)
def test_reciprocal_rank_hand_computed(ranked, k, expected):
    assert reciprocal_rank_at_k(ranked, ["a"], k) == pytest.approx(expected)


def test_reciprocal_rank_is_zero_past_the_cutoff():
    assert reciprocal_rank_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_score_uses_only_the_top_k_chunks_and_dedupes_within_them():
    row = score_question(question(), ["x.md", "x.md", "a.md", "b.md"], 2, 1.0)
    assert row.ranked_documents == ("x.md",)
    assert row.hit == 0.0
    assert row.first_relevant_chunk_rank == 3


def test_repeated_chunks_of_one_document_cannot_raise_recall():
    row = score_question(question(relevant=("a.md", "b.md")), ["a.md", "a.md", "a.md"], 3, 1.0)
    assert row.recall == 0.5


def test_aggregate_hand_computed():
    rows = [
        score_question(question("1"), ["a.md"], 5, 1.0),
        score_question(question("2"), ["x.md", "a.md"], 5, 1.0),
        score_question(question("3"), ["x.md"], 5, 1.0),
    ]
    result = aggregate(rows)
    assert result["hit_rate"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx(0.5)
    assert result["top_1_document"] == pytest.approx(1 / 3)


def test_aggregate_of_nothing_is_none_not_zero():
    assert aggregate([]) is None


def test_percentiles_are_observed_values():
    summary = latency_summary([float(value) for value in range(1, 101)])
    assert summary == {
        "count": 100,
        "mean_ms": 50.5,
        "p50_ms": 50.0,
        "p95_ms": 95.0,
        "max_ms": 100.0,
    }


def test_percentiles_of_one_value_and_of_none():
    assert latency_summary([7.0])["p95_ms"] == 7.0
    assert latency_summary([]) is None


class FakeRetriever:
    def __init__(self, sources):
        self.sources = sources

    def search(self, text, top_k):
        passages = tuple(
            RetrievedPassage(
                chunk_id=str(i),
                document_id="d",
                source=source,
                filename=source,
                page_number=None,
                chunk_index=i,
                text="text",
                score=0.5,
            )
            for i, source in enumerate(self.sources[:top_k])
        )
        return SearchResponse(
            query=text,
            top_k=top_k,
            source_filter=None,
            results=passages,
            embedding_time_ms=1.0,
            vector_query_time_ms=1.0,
            retrieval_time_ms=2.0,
        )


def test_unanswerable_questions_leave_the_denominators():
    questions = [question("a"), question("u", "unanswerable")]
    report = evaluate_retrieval(FakeRetriever(["a.md"]), questions, 5)
    assert report["overall"]["questions"] == 1
    assert report["overall"]["hit_rate"] == 1.0
    assert report["unanswerable_excluded"] == 1
    assert report["latency"]["count"] == 2


def test_passage_recall_is_reported_as_not_available():
    report = evaluate_retrieval(FakeRetriever(["a.md"]), [question()], 5)
    assert report["passage_level_recall"].startswith("N/A")


def test_report_breaks_results_down_by_category():
    questions = [question("a"), question("b", "paraphrased")]
    report = evaluate_retrieval(FakeRetriever(["x.md"]), questions, 5)
    assert set(report["by_category"]) == {"straightforward", "paraphrased"}


@pytest.mark.parametrize(
    "fields",
    [
        {"category": "unanswerable", "reference_answer": "x", "relevant_sources": ()},
        {"category": "unanswerable", "relevant_sources": ("a.md",), "reference_answer": None},
        {"reference_answer": None},
        {"relevant_sources": ()},
        {"supporting_passages": ()},
        {"required_sources": ("other.md",)},
        {"category": "multi_passage", "required_sources": ("a.md",)},
        {"required_sources": ("a.md", "b.md"), "relevant_sources": ("a.md", "b.md")},
    ],
)
def test_malformed_labels_are_rejected(fields):
    base = {
        "id": "q",
        "category": "straightforward",
        "question": "Q",
        "reference_answer": "A",
        "relevant_sources": ("a.md",),
        "required_sources": (),
        "supporting_passages": ("x",),
    }
    if fields.get("category") == "unanswerable":
        base["supporting_passages"] = ()
    with pytest.raises(ValueError):
        EvalQuestion(**{**base, **fields})


def document(source="a.md", text="The   quoted\ntext appears here.", digest="hash"):
    return Document(
        document_id="d",
        source_namespace="ns",
        source=source,
        filename=source,
        file_type="md",
        document_hash=digest,
        size_bytes=1,
        modified_at="2026-01-01T00:00:00Z",
        extractor="test",
        pages=(Page(raw_text=text, cleaned_text=text),),
    )


def test_valid_labels_pass_with_whitespace_tolerated():
    documents = [document()]
    validate_labels([question()], documents, corpus_fingerprint(documents))


def test_corpus_change_makes_labels_stale():
    lock = corpus_fingerprint([document()])
    with pytest.raises(EvaluationError, match="changed since"):
        validate_labels([question()], [document(digest="edited")], lock)


def test_fingerprint_ignores_document_order():
    a, b = document("a.md"), document("b.md")
    assert corpus_fingerprint([a, b]) == corpus_fingerprint([b, a])


def test_unknown_source_is_stale():
    documents = [document("b.md")]
    with pytest.raises(EvaluationError, match="unknown source"):
        validate_labels([question()], documents, corpus_fingerprint(documents))


def test_quote_missing_from_its_source_is_stale():
    documents = [document(text="Something else entirely.")]
    with pytest.raises(EvaluationError, match="not found"):
        validate_labels([question()], documents, corpus_fingerprint(documents))


def write_questions(path, items):
    path.write_text("\n".join(json.dumps(item) for item in items))


def label(id, category):
    fields = {
        "id": id,
        "category": category,
        "question": "Q",
        "reference_answer": "A",
        "relevant_sources": ["a.md"],
        "required_sources": [],
        "supporting_passages": ["x"],
    }
    if category == "unanswerable":
        fields.update(reference_answer=None, relevant_sources=[], supporting_passages=[])
    if category == "multi_passage":
        fields.update(relevant_sources=["a.md", "b.md"], required_sources=["a.md", "b.md"])
    return fields


def full_set():
    return [
        label(f"{category}{index}", category)
        for category, count in (
            ("straightforward", 10),
            ("paraphrased", 5),
            ("multi_passage", 5),
            ("unanswerable", 5),
        )
        for index in range(count)
    ]


def test_complete_question_file_loads(tmp_path):
    path = tmp_path / "q.jsonl"
    write_questions(path, full_set())
    assert len(load_questions(path)) == 25


def test_too_few_questions_in_a_category_fail(tmp_path):
    path = tmp_path / "q.jsonl"
    write_questions(path, full_set()[1:])
    with pytest.raises(EvaluationError, match="straightforward"):
        load_questions(path)


def test_duplicate_ids_fail(tmp_path):
    items = full_set()
    items[1]["id"] = items[0]["id"]
    path = tmp_path / "q.jsonl"
    write_questions(path, items)
    with pytest.raises(EvaluationError, match="unique"):
        load_questions(path)


def test_unreadable_files_fail_clearly(tmp_path):
    with pytest.raises(EvaluationError):
        load_questions(tmp_path / "missing.jsonl")
    with pytest.raises(EvaluationError):
        read_corpus_lock(tmp_path / "missing.json")
