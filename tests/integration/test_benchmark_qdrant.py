"""A small real benchmark on a real server. Measures nothing to publish; checks the protocol.

Requires: docker compose up -d qdrant. Needs no embedding or Ollama models.
"""

import pytest
from qdrant_client import QdrantClient

from rag_pipeline.config import Settings
from rag_pipeline.evaluation.benchmark import BenchmarkConfig, run_benchmark

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]


@pytest.fixture(scope="module")
def run():
    settings = Settings()
    client = QdrantClient(url=str(settings.qdrant_url), timeout=60, check_compatibility=False)
    before = {c.name for c in client.get_collections().collections}
    config = BenchmarkConfig(
        points=3000,
        queries=10,
        seed=42,
        top_k=5,
        warmup_queries=2,
        repetitions=1,
        hnsw_efs=(64,),
        readiness_timeout_seconds=120,
    )
    report = run_benchmark(client, config, settings.qdrant_collection, poll_seconds=0.2)
    return report, before, {c.name for c in client.get_collections().collections}


def test_index_really_exists_before_timings(run):
    report, _, _ = run
    assert report["ready"] is True
    assert report["indexed_fraction"] >= 0.95
    assert report["index_state"]["exact_points"] == 3000


def test_exact_mode_defines_ground_truth(run):
    report, _, _ = run
    assert report["modes"]["exact"]["recall"] == 1.0
    assert 0.0 <= report["modes"]["hnsw_ef_64"]["recall"] <= 1.0


def test_collection_is_removed_and_others_are_untouched(run):
    _, before, after = run
    assert before == after
