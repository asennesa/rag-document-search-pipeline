import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from rag_pipeline import assets
from rag_pipeline.assets import (
    EMBEDDING_EXPORT_REPOSITORY,
    prepare_embedding_tokenizer,
    resolve_embedding_tokenizer,
)
from rag_pipeline.config import Settings
from rag_pipeline.errors import AssetError
from rag_pipeline.identifiers import sha256_bytes
from rag_pipeline.tokenization import EmbeddingTokenizer


def write_manifest(path, files, repository=EMBEDDING_EXPORT_REPOSITORY):
    path.write_text(
        json.dumps(
            {
                "embedding_tokenizer": {
                    "repository": repository,
                    "revision": "abc",
                    "files": files,
                }
            }
        )
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(model_assets_path=tmp_path / "manifest.json")


def serve_cached(monkeypatch, path):
    monkeypatch.setattr(assets, "_cached_file", lambda *args: path)


def test_missing_manifest_gives_preparation_instruction(settings):
    with pytest.raises(AssetError, match="prepare_assets.py --embedding-tokenizer"):
        resolve_embedding_tokenizer(settings)


def test_malformed_manifest_is_reported(settings):
    settings.model_assets_path.write_text("{not json")
    with pytest.raises(AssetError, match="prepare_assets"):
        resolve_embedding_tokenizer(settings)


def test_uncached_file_gives_instruction_without_network(settings, monkeypatch):
    write_manifest(settings.model_assets_path, {"tokenizer.json": "hash"})
    serve_cached(monkeypatch, None)
    with pytest.raises(AssetError, match="missing.*prepare_assets"):
        resolve_embedding_tokenizer(settings)


def test_tampered_file_fails_checksum(settings, tmp_path, monkeypatch):
    file = tmp_path / "tokenizer.json"
    file.write_bytes(b"tampered")
    write_manifest(settings.model_assets_path, {"tokenizer.json": sha256_bytes(b"original")})
    serve_cached(monkeypatch, file)
    with pytest.raises(AssetError, match="checksum"):
        resolve_embedding_tokenizer(settings)


def test_other_repository_is_rejected(settings):
    write_manifest(settings.model_assets_path, {"tokenizer.json": "x"}, repository="other/model")
    with pytest.raises(AssetError, match="different repository"):
        resolve_embedding_tokenizer(settings)


def test_verified_file_resolves(settings, tmp_path, monkeypatch):
    file = tmp_path / "tokenizer.json"
    file.write_bytes(b"content")
    digest = sha256_bytes(b"content")
    write_manifest(settings.model_assets_path, {"tokenizer.json": digest})
    serve_cached(monkeypatch, file)
    asset = resolve_embedding_tokenizer(settings)
    assert (asset.tokenizer_file, asset.sha256) == (file, digest)


def test_download_failure_is_actionable_and_leaves_no_manifest(settings, monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(assets.HfApi, "model_info", offline)
    with pytest.raises(AssetError, match="connectivity"):
        prepare_embedding_tokenizer(settings)
    assert not settings.model_assets_path.exists()


def test_adapter_disables_truncation_saved_in_the_tokenizer_file(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=3)
    file = tmp_path / "tokenizer.json"
    tokenizer.save(str(file))
    assert len(Tokenizer.from_file(str(file)).encode("word " * 10).ids) == 3
    offsets = EmbeddingTokenizer(file).offsets("word " * 10)
    assert len(offsets) == 10
    assert offsets[1] == (5, 9)
