import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from rag_pipeline.errors import EvaluationError
from rag_pipeline.evaluation import benchmark
from rag_pipeline.evaluation.benchmark import (
    BenchmarkConfig,
    benchmark_collection_name,
    benchmark_data,
    is_ready,
    recall_against_exact,
    rotated_modes,
    run_benchmark,
    synthetic_vectors,
)


def config(**fields):
    return BenchmarkConfig(
        **{"points": 50, "queries": 5, "seed": 42, "top_k": 3, "dimensions": 8, **fields}
    )


def test_vectors_are_unit_length_float32_and_seeded():
    first = synthetic_vectors(10, 8, (42, 0))
    assert first.dtype == np.float32
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.array_equal(first, synthetic_vectors(10, 8, (42, 0)))
    assert not np.array_equal(first, synthetic_vectors(10, 8, (43, 0)))


def test_queries_are_held_out_from_stored_points():
    data, queries = benchmark_data(config())
    assert (np.round(data @ queries.T, 6) < 1.0).all()
    assert len(queries) == 5


def test_recall_hand_computed():
    assert recall_against_exact([[1, 2], [3, 4]], [[1, 9], [3, 4]], 2) == 0.75


def test_recall_ignores_order_within_the_top_k():
    assert recall_against_exact([[1, 2]], [[2, 1]], 2) == 1.0


def test_recall_needs_matching_query_sets():
    with pytest.raises(EvaluationError):
        recall_against_exact([[1]], [[1], [2]], 1)
    with pytest.raises(EvaluationError):
        recall_against_exact([], [], 1)


def test_modes_rotate_so_order_is_balanced():
    assert [rotated_modes(["a", "b", "c"], r)[0] for r in range(3)] == ["a", "b", "c"]


def test_collection_name_cannot_be_the_application_collection():
    assert benchmark_collection_name("x", "document_embeddings") == "rag_benchmark_x"
    with pytest.raises(EvaluationError):
        benchmark_collection_name("x", "rag_benchmark_x")


@pytest.mark.parametrize(
    "fields", [{"points": 2, "top_k": 3}, {"top_k": 0}, {"queries": 0}, {"repetitions": 0}]
)
def test_invalid_configuration_is_rejected(fields):
    with pytest.raises(EvaluationError):
        config(**fields)


@pytest.mark.parametrize(
    ("status", "optimizer", "exact", "indexed", "ready"),
    [
        ("green", "ok", 50, 50, True),
        ("green", "ok", 50, 48, True),
        ("green", "ok", 50, 47, False),
        ("yellow", "ok", 50, 50, False),
        ("green", "error", 50, 50, False),
        ("green", "ok", 49, 50, False),
        ("green", "ok", 50, None, False),
        ("green", "ok", 50, 0, False),
    ],
)
def test_readiness_needs_every_condition(status, optimizer, exact, indexed, ready):
    assert is_ready(status, optimizer, exact, indexed, config()) is ready


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_unready_index_gives_a_diagnostic_and_no_timings(monkeypatch):
    """Local mode never builds an index, which is exactly a failed readiness gate."""
    clock = Clock()
    monkeypatch.setattr(benchmark.time, "perf_counter", clock.perf_counter)
    client = QdrantClient(":memory:")
    report = run_benchmark(
        client,
        config(readiness_timeout_seconds=3.0),
        "document_embeddings",
        run_id="t",
        sleep=clock.advance,
    )
    assert report["ready"] is False
    assert report["modes"] is None
    assert not client.collection_exists("rag_benchmark_t")


def test_existing_collection_is_never_reused_or_removed():
    client = QdrantClient(":memory:")
    client.create_collection(
        "rag_benchmark_t",
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )
    with pytest.raises(EvaluationError, match="already exists"):
        run_benchmark(client, config(), "document_embeddings", run_id="t")
    assert client.collection_exists("rag_benchmark_t")


def test_failure_mid_run_still_removes_its_own_collection(monkeypatch):
    class FailingClient(QdrantClient):
        def upsert(self, *args, **kwargs):
            raise RuntimeError("upload failed")

    client = FailingClient(":memory:")
    with pytest.raises(RuntimeError):
        run_benchmark(client, config(), "document_embeddings", run_id="t")
    assert not client.collection_exists("rag_benchmark_t")


def test_all_modes_use_the_same_queries_and_exact_is_ground_truth(monkeypatch):
    seen = []

    def fake_search(_client, _name, query, k, mode):
        seen.append((mode, tuple(query)))
        return 1.0, [1, 2, 3] if mode == "exact" else [1, 2, 9]

    monkeypatch.setattr(benchmark, "_search", fake_search)
    queries = [[0.1], [0.2]]
    result = benchmark._measure(
        None, "name", queries, config(queries=2, hnsw_efs=(32,), warmup_queries=1, repetitions=2)
    )
    timed = seen[2:]  # Two warmup calls come first.
    assert len(timed) == 2 * 2 * 2
    assert {q for mode, q in timed if mode == "exact"} == {q for mode, q in timed if mode == 32}
    assert result["exact"]["recall"] == 1.0
    assert result["hnsw_ef_32"]["recall"] == pytest.approx(2 / 3)
    assert result["hnsw_ef_32"]["latency"]["count"] == 4
