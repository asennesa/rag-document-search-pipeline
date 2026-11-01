"""Document-level retrieval metrics.

The unit is the deduplicated document: two chunks from one policy count once. Relevance
labels are exhaustive, so Recall@K is meaningful here. Unanswerable questions have no
relevant documents and are excluded from every denominator.

- Hit Rate@K: share of questions with at least one relevant document in the top K.
- Recall@K: relevant documents in the top K, over all relevant documents, averaged.
- MRR@K: reciprocal rank of the first relevant document, zero for a miss, averaged.
"""

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from rag_pipeline.evaluation.dataset import EvalQuestion


def ranked_documents(sources: Iterable[str]) -> list[str]:
    """Deduplicate chunk sources, keeping each document at its best rank."""
    return list(dict.fromkeys(sources))


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Recall is undefined without relevant documents")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    for rank, source in enumerate(ranked[:k], start=1):
        if source in relevant:
            return 1 / rank
    return 0.0


def latency_summary(values: Sequence[float]) -> dict[str, float | int] | None:
    """Nearest-rank percentiles, so every reported value was actually observed."""
    if not values:
        return None
    ordered = sorted(values)

    def percentile(share: float) -> float:
        return ordered[max(math.ceil(share * len(ordered)) - 1, 0)]

    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


@dataclass(frozen=True)
class RetrievalRow:
    id: str
    category: str
    ranked_documents: tuple[str, ...]
    relevant: tuple[str, ...]
    hit: float
    recall: float
    reciprocal_rank: float
    first_relevant_chunk_rank: int | None
    retrieval_ms: float


def score_question(
    question: EvalQuestion, chunk_sources: Sequence[str], k: int, retrieval_ms: float
) -> RetrievalRow:
    """K counts chunks retrieved, as in search; documents are deduplicated within them."""
    ranked = ranked_documents(chunk_sources[:k])
    relevant = question.relevant_sources
    first = next((rank for rank, source in enumerate(chunk_sources, 1) if source in relevant), None)
    return RetrievalRow(
        id=question.id,
        category=question.category,
        ranked_documents=tuple(ranked),
        relevant=relevant,
        hit=hit_at_k(ranked, relevant, k),
        recall=recall_at_k(ranked, relevant, k),
        reciprocal_rank=reciprocal_rank_at_k(ranked, relevant, k),
        first_relevant_chunk_rank=first,
        retrieval_ms=retrieval_ms,
    )


def aggregate(rows: Sequence[RetrievalRow]) -> dict[str, float | int] | None:
    if not rows:
        return None
    return {
        "questions": len(rows),
        "hit_rate": statistics.fmean(row.hit for row in rows),
        "recall": statistics.fmean(row.recall for row in rows),
        "mrr": statistics.fmean(row.reciprocal_rank for row in rows),
        "top_1_document": statistics.fmean(
            1.0 if row.ranked_documents[:1] and row.ranked_documents[0] in row.relevant else 0.0
            for row in rows
        ),
    }


def evaluate_retrieval(retriever, questions: Sequence[EvalQuestion], k: int) -> dict:
    rows: list[RetrievalRow] = []
    latencies: list[float] = []
    excluded = 0
    for question in questions:
        response = retriever.search(question.question, k)
        latencies.append(response.retrieval_time_ms)
        if not question.answerable:
            excluded += 1
            continue
        sources = [passage.source for passage in response.results]
        rows.append(score_question(question, sources, k, response.retrieval_time_ms))
    categories = sorted({row.category for row in rows})
    return {
        "k": k,
        "unit": "deduplicated document",
        "overall": aggregate(rows),
        "by_category": {
            name: aggregate([row for row in rows if row.category == name]) for name in categories
        },
        "unanswerable_excluded": excluded,
        "passage_level_recall": "N/A: passage judgments are illustrative, not exhaustive",
        "latency": latency_summary(latencies),
        "rows": [asdict(row) for row in rows],
    }
