"""Exact versus approximate search on synthetic vectors, in its own collection.

These vectors test index mechanics, not language retrieval quality. A failed readiness
gate produces a diagnostic and no timings, because numbers from an unindexed collection
would describe a full scan.

`exact=False` permits approximate search and cannot prove a graph was traversed, so results
are labeled "ANN-permitted search on an indexed collection".
"""

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from qdrant_client import models

from rag_pipeline.errors import EvaluationError
from rag_pipeline.evaluation.retrieval_metrics import latency_summary

COLLECTION_PREFIX = "rag_benchmark_"
EXACT = "exact"
_BATCH_SIZE = 1000
_DISABLED_INDEXING = 0
_INDEXING_THRESHOLD_KB = 1024
_FULL_SCAN_THRESHOLD_KB = 10  # The lowest value Qdrant accepts.


@dataclass(frozen=True)
class BenchmarkConfig:
    points: int
    queries: int
    seed: int
    top_k: int
    dimensions: int = 384
    warmup_queries: int = 20
    repetitions: int = 3
    hnsw_efs: tuple[int, ...] = (32, 64, 128, 256)
    m: int = 16
    ef_construct: int = 100
    min_indexed_fraction: float = 0.95
    readiness_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.points < self.top_k:
            raise EvaluationError("Benchmark needs at least top_k points")
        if self.queries < 1 or self.repetitions < 1:
            raise EvaluationError("Benchmark needs at least one query and repetition")


def benchmark_collection_name(run_id: str, application_collection: str) -> str:
    name = f"{COLLECTION_PREFIX}{run_id}"
    if name == application_collection:
        raise EvaluationError("The benchmark must not use the application collection")
    return name


def synthetic_vectors(count: int, dimensions: int, seed: Sequence[int]) -> np.ndarray:
    """Unit-length float32 vectors, deterministic for a seed."""
    rng = np.random.default_rng(list(seed))
    vectors = rng.standard_normal((count, dimensions), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def benchmark_data(config: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Separate streams keep queries held out: none is a copy of a stored point."""
    data = synthetic_vectors(config.points, config.dimensions, (config.seed, 0))
    queries = synthetic_vectors(config.queries, config.dimensions, (config.seed, 1))
    return data, queries


def recall_against_exact(
    exact_ids: Sequence[Sequence[int]], approximate_ids: Sequence[Sequence[int]], k: int
) -> float:
    """Mean of |exact top K ∩ approximate top K| / K. K <= N, so K ids always exist."""
    if len(exact_ids) != len(approximate_ids) or not exact_ids:
        raise EvaluationError("Recall needs the same non-empty query set for both modes")
    shares = [
        len(set(exact[:k]) & set(approximate[:k])) / k
        for exact, approximate in zip(exact_ids, approximate_ids, strict=True)
    ]
    return float(np.mean(shares))


def rotated_modes(modes: Sequence[str | int], repetition: int) -> list[str | int]:
    """Change the order each repetition, so no mode always runs first on a warm cache."""
    offset = repetition % len(modes)
    return [*modes[offset:], *modes[:offset]]


def is_ready(status: str, optimizer: str, exact: int, indexed: int | None, config) -> bool:
    return (
        status == "green"
        and optimizer == "ok"
        and exact == config.points
        and indexed is not None
        and indexed >= config.min_indexed_fraction * config.points
    )


def _state(client, name: str) -> dict:
    info = client.get_collection(name)
    return {
        "status": str(getattr(info.status, "value", info.status)),
        "optimizer_status": str(getattr(info.optimizer_status, "value", info.optimizer_status)),
        "exact_points": client.count(name, exact=True).count,
        "indexed_vectors": info.indexed_vectors_count,
        "segments": info.segments_count,
        "hnsw_m": info.config.hnsw_config.m,
        "hnsw_ef_construct": info.config.hnsw_config.ef_construct,
        "full_scan_threshold_kb": info.config.hnsw_config.full_scan_threshold,
        "indexing_threshold_kb": info.config.optimizer_config.indexing_threshold,
    }


def _search(client, name: str, query: list[float], k: int, mode) -> tuple[float, list[int]]:
    params = (
        models.SearchParams(exact=True)
        if mode == EXACT
        else models.SearchParams(exact=False, hnsw_ef=mode)
    )
    started = time.perf_counter()
    response = client.query_points(
        collection_name=name,
        query=query,
        search_params=params,
        limit=k,
        with_payload=False,
        with_vectors=False,
    )
    elapsed = (time.perf_counter() - started) * 1000
    return elapsed, [point.id for point in response.points]


def _measure(client, name: str, queries: list[list[float]], config: BenchmarkConfig) -> dict:
    modes = [EXACT, *config.hnsw_efs]
    for mode in modes:  # Warmup results are discarded.
        for query in queries[: config.warmup_queries]:
            _search(client, name, query, config.top_k, mode)

    latencies: dict = {mode: [] for mode in modes}
    results: dict = {mode: [] for mode in modes}
    for repetition in range(config.repetitions):
        for mode in rotated_modes(modes, repetition):
            ids = []
            for query in queries:
                elapsed, found = _search(client, name, query, config.top_k, mode)
                latencies[mode].append(elapsed)
                ids.append(found)
            results[mode].append(ids)

    exact_ids = results[EXACT][0]
    measured = {}
    for mode in modes:
        recalls = [recall_against_exact(exact_ids, run, config.top_k) for run in results[mode]]
        label = EXACT if mode == EXACT else f"hnsw_ef_{mode}"
        measured[label] = {
            "latency": latency_summary(latencies[mode]),
            "recall": float(np.mean(recalls)),
        }
    return measured


def run_benchmark(
    client,
    config: BenchmarkConfig,
    application_collection: str,
    *,
    run_id: str | None = None,
    memory_probe: Callable[[], str | None] = lambda: None,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 1.0,
) -> dict:
    """Create, measure, and remove one fresh collection. Removes only what it created."""
    run_id = run_id or uuid.uuid4().hex[:12]
    name = benchmark_collection_name(run_id, application_collection)
    if client.collection_exists(name):
        raise EvaluationError("Benchmark collection already exists; nothing was changed")
    data, query_array = benchmark_data(config)
    queries = query_array.tolist()

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=config.dimensions, distance=models.Distance.COSINE),
        shard_number=1,
        hnsw_config=models.HnswConfigDiff(
            m=config.m,
            ef_construct=config.ef_construct,
            full_scan_threshold=_FULL_SCAN_THRESHOLD_KB,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=_DISABLED_INDEXING, default_segment_number=1
        ),
    )
    try:
        memory = {"before_upload": memory_probe()}
        started = time.perf_counter()
        for start in range(0, config.points, _BATCH_SIZE):
            stop = min(start + _BATCH_SIZE, config.points)
            client.upsert(
                collection_name=name,
                points=models.Batch(
                    ids=list(range(start, stop)), vectors=data[start:stop].tolist()
                ),
                wait=True,
            )
        upload_seconds = time.perf_counter() - started
        memory["after_upload"] = memory_probe()

        started = time.perf_counter()
        client.update_collection(
            collection_name=name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=_INDEXING_THRESHOLD_KB
            ),
        )
        state = _state(client, name)
        ready = False
        while time.perf_counter() - started < config.readiness_timeout_seconds:
            state = _state(client, name)
            if is_ready(
                state["status"],
                state["optimizer_status"],
                state["exact_points"],
                state["indexed_vectors"],
                config,
            ):
                ready = True
                break
            sleep(poll_seconds)
        readiness_seconds = time.perf_counter() - started
        memory["after_indexing"] = memory_probe()

        report = {
            "run_id": run_id,
            "config": asdict(config),
            "collection": name,
            "upload_seconds": upload_seconds,
            "index_readiness_wall_seconds": readiness_seconds,
            "index_state": state,
            "indexed_fraction": (state["indexed_vectors"] or 0) / config.points,
            "memory_docker_stats": memory,
            "ready": ready,
            "comparison": "exact search versus ANN-permitted search on an indexed collection",
        }
        report["modes"] = _measure(client, name, queries, config) if ready else None
        return report
    finally:
        client.delete_collection(name)
