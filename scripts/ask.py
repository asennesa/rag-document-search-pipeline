"""Ask a question and get an answer from local documents through Ollama.

Exit codes: 0 answered or refused, 1 an answer that failed validation or was cut off,
2 no answer could be generated. Passages are still shown when generation fails.
"""

import argparse
import sys
import textwrap

from rag_pipeline.cli import run_cli
from rag_pipeline.config import Settings
from rag_pipeline.errors import PipelineError
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.models import RagResponse
from rag_pipeline.rag.rag_service import load_rag_service

_SUCCESS = {"answered", "insufficient_evidence", "no_context"}
_UNRELIABLE = {"citation_invalid", "incomplete"}


def print_response(response: RagResponse) -> None:
    print(f"Question: {response.question}")
    print(f"Status: {response.status}")
    if response.message:
        print(f"Note: {response.message}")
    if response.answer:
        print(f"\nGenerated answer:\n{textwrap.fill(response.answer, width=88)}")
    cited = response.citation.cited_ids if response.citation else ()
    if response.status == "answered":
        print("\nCited sources (the IDs are valid; this does not prove the claims):")
        for source in response.sources:
            if source.source_id in cited:
                page = f", page {source.passage.page_number}" if source.passage.page_number else ""
                print(f"  [{source.source_id}] {source.passage.source}{page}")
    print("\nRetrieved passages (source data, not generated):")
    for passage in response.retrieved:
        print(f"  {passage.score:.4f} {passage.source}: {textwrap.shorten(passage.text, 110)}")
    print(
        f"\nTimings: retrieval {response.retrieval_time_ms:.1f} ms, "
        f"context {response.context_time_ms:.1f} ms, "
        f"generation {response.generation_time_ms:.1f} ms, "
        f"total {response.total_time_ms:.1f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="natural-language question")
    parser.add_argument("--top-k", type=int, help="number of passages to retrieve")
    parser.add_argument("--source", help="only use this relative source path")
    parser.add_argument("--json", action="store_true", help="print the full structured response")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    try:
        response = load_rag_service(settings).ask(args.question, args.top_k, args.source)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(response.model_dump_json(indent=2))
    else:
        print_response(response)
    if response.status in _SUCCESS:
        return 0
    return 1 if response.status in _UNRELIABLE else 2


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
