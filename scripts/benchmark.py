"""Compare exact and ANN-permitted search on synthetic vectors in a separate collection.

Needs Qdrant only. This is optional and never runs at application startup.
Exit codes: 0 measured, 1 index never became ready (diagnostic only), 2 could not run.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from rag_pipeline.cli import run_cli
from rag_pipeline.config import Settings
from rag_pipeline.errors import PipelineError
from rag_pipeline.evaluation.benchmark import BenchmarkConfig, run_benchmark


def docker_memory() -> str | None:
    """The Qdrant container only, not host or Docker VM memory."""
    try:
        done = subprocess.run(
            ["docker", "compose", "stats", "--no-stream", "--format", "{{.MemUsage}}", "qdrant"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()

    settings = Settings()
    client = QdrantClient(url=str(settings.qdrant_url), timeout=60, check_compatibility=False)
    try:
        config = BenchmarkConfig(
            points=args.points,
            queries=args.queries,
            seed=args.seed,
            top_k=args.top_k,
            m=settings.qdrant_hnsw_m,
            ef_construct=settings.qdrant_hnsw_ef_construct,
        )
        report = run_benchmark(
            client, config, settings.qdrant_collection, memory_probe=docker_memory
        )
        server = client.info()
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (ApiException, OSError) as error:
        print(f"error: Qdrant request failed: {type(error).__name__}", file=sys.stderr)
        return 2

    report["environment"] = {
        "qdrant_server": server.version,
        "qdrant_commit": server.commit,
        "machine": platform.machine(),
        "system": platform.platform(),
        "note": "Single serial client on a laptop with Qdrant in a Docker VM",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.output / f"benchmark-{args.points}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    state = report["index_state"]
    print(f"Points {args.points}, queries {args.queries}, K {args.top_k}, seed {args.seed}")
    print(
        f"Upload {report['upload_seconds']:.1f} s; index readiness wall "
        f"{report['index_readiness_wall_seconds']:.1f} s"
    )
    print(
        f"Indexed {state['indexed_vectors']} ({report['indexed_fraction']:.1%}), "
        f"segments {state['segments']}, status {state['status']}/{state['optimizer_status']}"
    )
    print(f"Container memory (docker stats): {report['memory_docker_stats']}")
    if not report["ready"]:
        print("Index was not ready. No timings were taken, and no speedup can be claimed.")
        print(f"Report: {path}")
        return 1
    print("Comparison: exact search versus ANN-permitted search on an indexed collection")
    for mode, values in report["modes"].items():
        latency = values["latency"]
        print(
            f"  {mode:<13} recall {values['recall']:.4f}  mean {latency['mean_ms']:6.2f} ms  "
            f"p50 {latency['p50_ms']:6.2f} ms  p95 {latency['p95_ms']:6.2f} ms"
        )
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
