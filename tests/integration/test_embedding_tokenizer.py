"""Checks against the real prepared tokenizer files. No ONNX weights are loaded.

Requires: uv run --locked python scripts/prepare_assets.py --embedding-tokenizer
Run from the repository root.
"""

import shutil
from pathlib import Path

import pytest
from fastembed import TextEmbedding
from fastembed.common.preprocessor_utils import load_tokenizer

from rag_pipeline.assets import (
    EMBEDDING_EXPORT_REPOSITORY,
    EMBEDDING_TOKENIZER_FILES,
    embedding_assets_id,
    resolve_embedding_tokenizer,
)
from rag_pipeline.config import Settings
from rag_pipeline.ingestion.chunker import Chunker
from rag_pipeline.ingestion.ingestion_pipeline import prepare_documents
from rag_pipeline.tokenization import EmbeddingTokenizer

pytestmark = [pytest.mark.integration, pytest.mark.model]

HARD_TEXTS = [
    "Plain words about annual leave. " * 300,
    "Überlänge naïve Çedilla résumé. " * 200,
    "日本語のテキストと中文文本。" * 150,
    "😀 emoji 🧪 between words " * 150,
    "x" * 6000,
    "supercalifragilisticexpialidocious" * 200,
    "[SEP] [CLS] [PAD] [MASK] [UNK] literal tokens " * 80,
    "line one\nline two\n\nnext paragraph\n" * 150,
    "a\u200bb\u00adc\ufe0fd " * 300,
]


@pytest.fixture(scope="module")
def settings():
    return Settings()


@pytest.fixture(scope="module")
def asset(settings):
    return resolve_embedding_tokenizer(settings)


@pytest.fixture(scope="module")
def tokenizer(asset):
    return EmbeddingTokenizer(asset.tokenizer_file)


@pytest.fixture
def fastembed_tokenizer(asset, tmp_path):
    """The tokenizer exactly as FastEmbed configures it, truncation included."""
    for name in EMBEDDING_TOKENIZER_FILES:
        shutil.copy(Path(asset.tokenizer_file).parent / name, tmp_path / name)
    return load_tokenizer(tmp_path)[0]


def test_manifest_repository_is_the_export_fastembed_selects():
    [description] = [
        model
        for model in TextEmbedding._list_supported_models()
        if model.model == "BAAI/bge-small-en-v1.5"
    ]
    assert description.sources.hf == EMBEDDING_EXPORT_REPOSITORY


def test_counts_are_not_capped_while_fastembed_truncates(tokenizer, fastembed_tokenizer):
    text = "word " * 2000
    assert len(tokenizer.offsets(text)) == 2000
    assert len(fastembed_tokenizer.encode(text).ids) == 512


def test_special_tokens_are_cls_and_sep(tokenizer, fastembed_tokenizer):
    assert tokenizer.special_token_count == 2
    assert fastembed_tokenizer.encode("word").tokens[0] == "[CLS]"
    assert fastembed_tokenizer.encode("word").tokens[-1] == "[SEP]"


@pytest.mark.parametrize("text", HARD_TEXTS)
def test_default_chunks_reach_the_encoder_whole(settings, tokenizer, fastembed_tokenizer, text):
    chunker = Chunker(
        tokenizer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_input_tokens=settings.embedding_max_tokens,
    )
    spans = chunker.split_text(text.strip())
    covered = bytearray(len(text.strip()))
    for span in spans:
        encoded = fastembed_tokenizer.encode(span.text)
        # FastEmbed truncating anything would show as a shorter or capped encoding.
        assert len(encoded.ids) == span.token_count <= settings.embedding_max_tokens
        assert len(tokenizer.offsets(span.text)) <= settings.chunk_size
        assert text.strip()[span.start_char : span.end_char] == span.text
        covered[span.start_char : span.end_char] = b"\x01" * (span.end_char - span.start_char)
    assert all(covered[i] or char.isspace() for i, char in enumerate(text.strip()))


def test_largest_configurable_chunk_reaches_exactly_the_model_limit(tokenizer, fastembed_tokenizer):
    chunker = Chunker(tokenizer, chunk_size=510, chunk_overlap=0, max_input_tokens=512)
    spans = chunker.split_text("word " * 1500)
    assert max(span.token_count for span in spans) == 512
    assert all(len(fastembed_tokenizer.encode(span.text).ids) == span.token_count for span in spans)


def test_sample_corpus_prepares_cleanly_with_defaults(settings, asset, tokenizer):
    assets_id = embedding_assets_id(settings)
    first = prepare_documents(settings, tokenizer, assets_id)
    second = prepare_documents(settings, tokenizer, assets_id)
    assert first.failures == ()
    assert len(first.documents) == 12
    assert first.documents == second.documents
    chunks = [chunk for prepared in first.documents for chunk in prepared.chunks]
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.token_count <= settings.chunk_size + 2 for chunk in chunks)
