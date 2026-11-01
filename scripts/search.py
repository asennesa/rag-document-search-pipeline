"""Search the indexed documents. Needs Qdrant and the embedding model, not Ollama.

Exit codes: 0 search ran (including no matches), 2 the search could not run.
"""

import argparse
import sys
import textwrap

from rag_pipeline.config import Settings
from rag_pipeline.errors import PipelineError
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.models import SearchResponse
from rag_pipeline.retrieval.semantic_search import load_search_service


def print_response(response: SearchResponse) -> None:
    print(f"Question: {response.query}")
    if response.source_filter:
        print(f"Source filter: {response.source_filter}")
    if not response.results:
        print("No matching passages.")
    for rank, passage in enumerate(response.results, start=1):
        page = f", page {passage.page_number}" if passage.page_number else ""
        print(f"\n{rank}. score {passage.score:.4f} | {passage.source}{page}")
        print(f"   chunk {passage.chunk_index}, id {passage.chunk_id}")
        print(textwrap.indent(textwrap.shorten(passage.text, width=300), "   "))
    print(
        f"\nTimings: embedding {response.embedding_time_ms:.1f} ms, "
        f"vector query {response.vector_query_time_ms:.1f} ms, "
        f"total {response.retrieval_time_ms:.1f} ms"
    )
    print("Scores rank passages for this question. They are not confidence or probability.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="natural-language question")
    parser.add_argument("--top-k", type=int, help="number of passages (default from settings)")
    parser.add_argument("--source", help="only search this relative source path")
    parser.add_argument("--json", action="store_true", help="print the full structured response")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    try:
        response = load_search_service(settings).search(args.question, args.top_k, args.source)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(response.model_dump_json(indent=2))
    else:
        print_response(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
