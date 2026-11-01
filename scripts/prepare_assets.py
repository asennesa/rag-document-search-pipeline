"""Download model assets explicitly. Nothing else in the project downloads anything."""

import argparse
import sys

from rag_pipeline.assets import (
    prepare_embedding_model,
    prepare_embedding_tokenizer,
    prepare_generation_tokenizer,
    resolve_embedding_model,
    resolve_embedding_tokenizer,
    resolve_generation_tokenizer,
)
from rag_pipeline.cli import run_cli
from rag_pipeline.config import Settings
from rag_pipeline.errors import PipelineError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-tokenizer",
        action="store_true",
        help="fetch the embedding tokenizer files (no ONNX weights) and pin them",
    )
    parser.add_argument(
        "--embedding-model",
        action="store_true",
        help="fetch the ONNX weights, about 67 MB, from the tokenizer's pinned commit",
    )
    parser.add_argument(
        "--generation-tokenizer",
        action="store_true",
        help="fetch the Qwen tokenizer used to budget prompts; the model comes from Ollama",
    )
    args = parser.parse_args()
    if not (args.embedding_tokenizer or args.embedding_model or args.generation_tokenizer):
        parser.error("choose --embedding-tokenizer, --embedding-model, or --generation-tokenizer")

    settings = Settings()
    try:
        if args.generation_tokenizer:
            revision = prepare_generation_tokenizer(settings)
            resolve_generation_tokenizer(settings)
            print(f"Generation tokenizer ready at revision {revision}")
            if not (args.embedding_tokenizer or args.embedding_model):
                return 0
        if args.embedding_model:
            revision = prepare_embedding_model(settings)
            model = resolve_embedding_model(settings)
            print(f"Embedding model ready: sha256 {model.sha256}")
        else:
            revision = prepare_embedding_tokenizer(settings)
        asset = resolve_embedding_tokenizer(settings)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Embedding assets ready at revision {revision}")
    print(f"tokenizer.json sha256 {asset.sha256}")
    print(f"Manifest: {settings.model_assets_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
