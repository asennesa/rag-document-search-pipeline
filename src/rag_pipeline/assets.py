"""Explicit model asset preparation and verification.

Only the `prepare_*` functions touch the network. Normal runs use the commit and hashes
recorded in the tracked manifest and fail with an instruction if they do not match.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

from rag_pipeline.config import Settings
from rag_pipeline.errors import AssetError
from rag_pipeline.identifiers import make_processing_fingerprint, sha256_bytes

# The ONNX export FastEmbed selects for this model. All files must come from it.
EMBEDDING_EXPORT_REPOSITORY = "Qdrant/bge-small-en-v1.5-onnx-Q"
EMBEDDING_MODEL_FILE = "model_optimized.onnx"
EMBEDDING_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)
# Used only to count prompt tokens; inference itself runs inside Ollama.
GENERATION_TOKENIZER_REPOSITORY = "Qwen/Qwen3-4B-Instruct-2507"
GENERATION_TOKENIZER_FILE = "tokenizer.json"
_GENERATION_HINT = "Run: uv run --locked python scripts/prepare_assets.py --generation-tokenizer"
_TOKENIZER_HINT = "Run: uv run --locked python scripts/prepare_assets.py --embedding-tokenizer"
_MODEL_HINT = "Run: uv run --locked python scripts/prepare_assets.py --embedding-model"


@dataclass(frozen=True)
class EmbeddingTokenizerAsset:
    tokenizer_file: Path
    sha256: str  # Hash of tokenizer.json.


@dataclass(frozen=True)
class EmbeddingModelAsset:
    model_dir: Path  # Pinned snapshot holding the ONNX file and its tokenizer files.
    sha256: str  # Hash of the ONNX file.


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AssetError(f"Asset manifest is unreadable. {_TOKENIZER_HINT}") from error
    if not isinstance(manifest, dict):
        raise AssetError(f"Asset manifest is malformed. {_TOKENIZER_HINT}")
    return manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _cached_file(settings: Settings, revision: str, filename: str) -> Path | None:
    cached = try_to_load_from_cache(
        EMBEDDING_EXPORT_REPOSITORY,
        filename,
        cache_dir=settings.embedding_cache_dir,
        revision=revision,
    )
    return Path(cached) if isinstance(cached, str) else None


def _download_hashes(settings: Settings, revision: str, filenames: tuple[str, ...]) -> dict:
    return {
        filename: _file_hash(
            Path(
                hf_hub_download(
                    EMBEDDING_EXPORT_REPOSITORY,
                    filename,
                    revision=revision,
                    cache_dir=settings.embedding_cache_dir,
                )
            )
        )
        for filename in filenames
    }


def prepare_embedding_tokenizer(settings: Settings) -> str:
    """Download tokenizer files at a pinned commit and return that commit.

    The first run pins the current upstream commit. Later runs reuse the recorded commit,
    so assets never move unnoticed; deleting the manifest entry is a deliberate upgrade.
    """
    manifest = _read_manifest(settings.model_assets_path)
    pinned = manifest.get("embedding_tokenizer")
    if pinned and pinned.get("repository") != EMBEDDING_EXPORT_REPOSITORY:
        raise AssetError("Asset manifest names a different embedding repository")
    try:
        revision = (
            pinned["revision"] if pinned else HfApi().model_info(EMBEDDING_EXPORT_REPOSITORY).sha
        )
        hashes = _download_hashes(settings, revision, EMBEDDING_TOKENIZER_FILES)
    except Exception as error:  # The Hub client raises many transport-specific types.
        raise AssetError("Could not download tokenizer files; check connectivity") from error
    if pinned and pinned.get("files") != hashes:
        raise AssetError("Downloaded tokenizer files do not match the manifest hashes")

    manifest["embedding_tokenizer"] = {
        "logical_model": settings.embedding_model,
        "repository": EMBEDDING_EXPORT_REPOSITORY,
        "revision": revision,
        "files": hashes,
    }
    _write_manifest(settings.model_assets_path, manifest)
    return revision


def prepare_embedding_model(settings: Settings) -> str:
    """Download the ONNX weights from the same commit as the tokenizer and return it."""
    revision = prepare_embedding_tokenizer(settings)
    manifest = _read_manifest(settings.model_assets_path)
    pinned = manifest.get("embedding_model")
    if pinned and pinned.get("revision") != revision:
        raise AssetError("Manifest pins the model and tokenizer to different commits")
    try:
        hashes = _download_hashes(settings, revision, (EMBEDDING_MODEL_FILE,))
    except Exception as error:  # The Hub client raises many transport-specific types.
        raise AssetError("Could not download the embedding model; check connectivity") from error
    if pinned and pinned.get("files") != hashes:
        raise AssetError("Downloaded embedding model does not match the manifest hash")

    manifest["embedding_model"] = {
        "logical_model": settings.embedding_model,
        "repository": EMBEDDING_EXPORT_REPOSITORY,
        "revision": revision,
        "files": hashes,
    }
    _write_manifest(settings.model_assets_path, manifest)
    return revision


def _verified_files(settings: Settings, entry: dict, hint: str) -> dict[str, Path]:
    """Check every file of a manifest entry against its recorded hash."""
    if entry.get("repository") != EMBEDDING_EXPORT_REPOSITORY:
        raise AssetError(f"Asset manifest names a different repository. {hint}")
    try:
        revision, expected = entry["revision"], entry["files"]
    except KeyError as error:
        raise AssetError(f"Asset manifest is incomplete. {hint}") from error
    verified = {}
    for filename, expected_hash in expected.items():
        path = _cached_file(settings, revision, filename)
        if path is None:
            raise AssetError(f"Asset file {filename} is missing. {hint}")
        if _file_hash(path) != expected_hash:
            raise AssetError(f"Asset file {filename} fails its checksum. {hint}")
        verified[filename] = path
    return verified


def resolve_embedding_tokenizer(settings: Settings) -> EmbeddingTokenizerAsset:
    """Locate verified local tokenizer files without any network access."""
    entry = _read_manifest(settings.model_assets_path).get("embedding_tokenizer")
    if not entry:
        raise AssetError(f"Embedding tokenizer is not prepared. {_TOKENIZER_HINT}")
    files = _verified_files(settings, entry, _TOKENIZER_HINT)
    if "tokenizer.json" not in files:
        raise AssetError(f"Asset manifest is incomplete. {_TOKENIZER_HINT}")
    return EmbeddingTokenizerAsset(files["tokenizer.json"], entry["files"]["tokenizer.json"])


def resolve_embedding_model(settings: Settings) -> EmbeddingModelAsset:
    """Locate the verified local ONNX snapshot without any network access."""
    manifest = _read_manifest(settings.model_assets_path)
    entry = manifest.get("embedding_model")
    tokenizer_entry = manifest.get("embedding_tokenizer")
    if not entry or not tokenizer_entry:
        raise AssetError(f"Embedding model is not prepared. {_MODEL_HINT}")
    if entry.get("revision") != tokenizer_entry.get("revision"):
        raise AssetError(f"Model and tokenizer come from different commits. {_MODEL_HINT}")
    _verified_files(settings, tokenizer_entry, _MODEL_HINT)
    files = _verified_files(settings, entry, _MODEL_HINT)
    if EMBEDDING_MODEL_FILE not in files:
        raise AssetError(f"Asset manifest is incomplete. {_MODEL_HINT}")
    return EmbeddingModelAsset(
        files[EMBEDDING_MODEL_FILE].parent, entry["files"][EMBEDDING_MODEL_FILE]
    )


def prepare_generation_tokenizer(settings: Settings) -> str:
    """Pin the Qwen tokenizer file. Like the others, a recorded commit never moves."""
    manifest = _read_manifest(settings.model_assets_path)
    pinned = manifest.get("generation_tokenizer")
    try:
        revision = (
            pinned["revision"]
            if pinned
            else HfApi().model_info(GENERATION_TOKENIZER_REPOSITORY).sha
        )
        path = hf_hub_download(
            GENERATION_TOKENIZER_REPOSITORY,
            GENERATION_TOKENIZER_FILE,
            revision=revision,
            cache_dir=settings.generation_cache_dir,
        )
        digest = _file_hash(Path(path))
    except Exception as error:  # The Hub client raises many transport-specific types.
        raise AssetError("Could not download the generation tokenizer") from error
    if pinned and pinned.get("sha256") != digest:
        raise AssetError("Downloaded generation tokenizer does not match the manifest hash")
    manifest["generation_tokenizer"] = {
        "repository": GENERATION_TOKENIZER_REPOSITORY,
        "revision": revision,
        "sha256": digest,
    }
    _write_manifest(settings.model_assets_path, manifest)
    return revision


def resolve_generation_tokenizer(settings: Settings) -> Path:
    """Locate the verified local tokenizer without any network access."""
    entry = _read_manifest(settings.model_assets_path).get("generation_tokenizer")
    if not entry:
        raise AssetError(f"Generation tokenizer is not prepared. {_GENERATION_HINT}")
    cached = try_to_load_from_cache(
        GENERATION_TOKENIZER_REPOSITORY,
        GENERATION_TOKENIZER_FILE,
        cache_dir=settings.generation_cache_dir,
        revision=entry.get("revision"),
    )
    if not isinstance(cached, str):
        raise AssetError(f"Generation tokenizer file is missing. {_GENERATION_HINT}")
    if _file_hash(Path(cached)) != entry.get("sha256"):
        raise AssetError(f"Generation tokenizer fails its checksum. {_GENERATION_HINT}")
    return Path(cached)


def embedding_assets_id(settings: Settings) -> str:
    """Identify the vector space from the manifest alone, so no weights are needed."""
    manifest = _read_manifest(settings.model_assets_path)
    tokenizer, model = manifest.get("embedding_tokenizer"), manifest.get("embedding_model")
    if not tokenizer or not model:
        raise AssetError(f"Embedding assets are not recorded. {_MODEL_HINT}")
    return make_processing_fingerprint(
        model=settings.embedding_model,
        tokenizer_files=tokenizer.get("files"),
        model_files=model.get("files"),
        revision=model.get("revision"),
    )
