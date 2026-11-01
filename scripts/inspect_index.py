"""Show observed collection and index state. This reads only; it never changes anything."""

import hashlib
import sys

from rag_pipeline.config import Settings
from rag_pipeline.errors import PipelineError
from rag_pipeline.vector_store.qdrant_store import CollectionStats, connect_store


def interpret(stats: CollectionStats) -> str:
    if not stats.indexed_vectors:
        return (
            "No vectors are in an HNSW index. Queries run as a full scan, which is expected "
            "for a collection below the indexing threshold. HNSW is configured, not in use."
        )
    return (
        "Some vectors are in an HNSW index, so indexed search is available. This does not "
        "prove that any particular query used it; Qdrant plans each query separately."
    )


def main() -> int:
    settings = Settings()
    try:
        store = connect_store(settings)
        if not store.collection_exists():
            print(f"Collection {settings.qdrant_collection} does not exist yet.", file=sys.stderr)
            return 2
        stats = store.stats()
        ids = store.point_ids()
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    def shown(value: object) -> object:
        return "not reported" if value is None else value

    print(f"Collection: {stats.name}")
    print(f"Status: {stats.status}")
    print(f"Optimizer status: {stats.optimizer_status}")
    print(f"Points (exact count): {stats.exact_points}")
    print(f"Points (approximate counter): {shown(stats.approximate_points)}")
    print(f"Indexed vectors: {shown(stats.indexed_vectors)}")
    print(f"Segments: {shown(stats.segments)}")
    print(f"Vector dimensions: {stats.dimensions}")
    print(f"Distance: {stats.distance}")
    print(f"HNSW m: {stats.hnsw_m}")
    print(f"HNSW ef_construct: {stats.hnsw_ef_construct}")
    print(f"Query hnsw_ef (application setting): {settings.qdrant_hnsw_ef}")
    print(f"Full scan threshold: {stats.full_scan_threshold_kb} KB of vectors")
    print(f"Indexing threshold: {shown(stats.indexing_threshold_kb)} KB of vectors")
    print(f"Payload indexes: {', '.join(stats.payload_indexes) or 'none'}")
    print(f"Point ID digest: {hashlib.sha256(chr(10).join(ids).encode()).hexdigest()}")
    print(f"Interpretation: {interpret(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
