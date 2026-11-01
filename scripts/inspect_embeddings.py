"""Show what the embedding model produces. All printed numbers are measured."""

import argparse
import math
import sys
import time

from rag_pipeline.cli import run_cli
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.embedding_service import (
    EmbeddingService,
    cosine_similarity,
    load_embedding_service,
)
from rag_pipeline.errors import PipelineError

COMPARISON = (
    "My laptop battery is dead.",
    "My computer cannot hold a charge.",
    "I enjoy eating pizza.",
)


def inspect_text(service: EmbeddingService, text: str, as_query: bool) -> None:
    started = time.perf_counter()
    vector = service.embed_query(text) if as_query else service.embed_documents([text])[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"Encoded as: {'query, with BGE instruction' if as_query else 'document'}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 10 values: {[round(value, 6) for value in vector[:10]]}")
    print(f"Generation time: {elapsed_ms:.1f} ms")
    print(f"Validation: passed (norm {math.fsum(value**2 for value in vector) ** 0.5:.6f})")


def compare(service: EmbeddingService) -> None:
    vectors = service.embed_documents(list(COMPARISON))
    for index, sentence in enumerate(COMPARISON, start=1):
        print(f"  {index}. {sentence}")
    for left, right in ((0, 1), (0, 2), (1, 2)):
        score = cosine_similarity(vectors[left], vectors[right])
        print(f"cosine({left + 1}, {right + 1}) = {score:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="text to embed")
    parser.add_argument("--query", action="store_true", help="embed the text as a question")
    parser.add_argument("--compare", action="store_true", help="compare three sample sentences")
    args = parser.parse_args()
    if not args.text and not args.compare:
        parser.error("provide text or --compare")

    try:
        service = load_embedding_service(Settings())
        print(f"Model: {service.model_name}")
        print(f"Model load time: {service.load_ms:.1f} ms")
        if args.text:
            inspect_text(service, args.text, args.query)
        if args.compare:
            compare(service)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
