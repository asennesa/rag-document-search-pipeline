"""The whole workflow through its public interfaces, in a disposable collection.

Requires: prepared assets and docker compose up -d qdrant. Generation steps also need
Ollama and are skipped as BLOCKED, not passed, without it.
"""

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import ollama
import pytest
import streamlit as st
from qdrant_client import QdrantClient, models
from streamlit.testing.v1 import AppTest

from rag_pipeline.config import Settings

pytestmark = [pytest.mark.integration, pytest.mark.model]

REPOSITORY = Path(__file__).parents[2]
SAMPLES = REPOSITORY / "data" / "sample_docs"


def command(*arguments, **variables):
    environment = {**os.environ, "LOG_LEVEL": "ERROR", **variables}
    return subprocess.run(
        [sys.executable, "-W", "ignore", *arguments],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )


class Workspace:
    def __init__(self, root):
        self.documents = root / "documents"
        shutil.copytree(SAMPLES, self.documents)
        self.collection = f"test_phase9_{uuid.uuid4().hex}"
        self.variables = {
            "QDRANT_COLLECTION": self.collection,
            "DOCUMENTS_DIR": str(self.documents),
            "UPLOADS_DIR": str(root / "uploads"),
            "INGESTION_LOCK_DIR": str(root / "locks"),
        }

    def run(self, *arguments, **variables):
        return command(*arguments, **{**self.variables, **variables})


@pytest.fixture(scope="module")
def client():
    return QdrantClient(url=str(Settings().qdrant_url), check_compatibility=False)


@pytest.fixture
def workspace(tmp_path, client):
    built = Workspace(tmp_path)
    yield built
    if client.collection_exists(built.collection):
        client.delete_collection(built.collection)


def has_ollama():
    settings = Settings()
    try:
        ollama.Client(host=str(settings.ollama_base_url)).show(settings.ollama_model)
    except (ConnectionError, httpx.HTTPError, ollama.ResponseError):
        return False
    return True


def test_full_workflow(workspace, monkeypatch, tmp_path):
    first = workspace.run("scripts/ingest.py")
    assert first.returncode == 0, first.stderr
    assert "12 synced" in first.stdout

    repeat = workspace.run("scripts/ingest.py")
    assert "12 unchanged" in repeat.stdout
    assert "Vectors upserted: 0" in repeat.stdout

    evaluated = workspace.run(
        "scripts/evaluate.py", "--mode", "retrieval", "--output", str(tmp_path)
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert "Hit Rate@K" in evaluated.stdout

    policy = workspace.documents / "annual_leave_policy.md"
    policy.write_text("# Annual Leave\n\nEmployees receive exactly 31 days of leave.\n")
    updated = workspace.run("scripts/ingest.py")
    assert "1 synced, 11 unchanged" in updated.stdout
    assert "Obsolete deleted: 2" in updated.stdout

    found = workspace.run(
        "scripts/search.py",
        "How many days of leave?",
        "--source",
        "annual_leave_policy.md",
        "--json",
    )
    assert found.returncode == 0, found.stderr
    results = json.loads(found.stdout)["results"]
    assert len(results) == 1 and "31 days" in results[0]["text"]

    inspected = workspace.run("scripts/inspect_index.py")
    assert "Points (exact count): 23" in inspected.stdout
    assert "HNSW is configured, not in use" in inspected.stdout

    policy.unlink()
    previewed = workspace.run("scripts/ingest.py", "--prune-missing", "--dry-run")
    assert "Would prune: 1" in previewed.stdout
    pruned = workspace.run("scripts/ingest.py", "--prune-missing")
    assert "pruned  annual_leave_policy.md" in pruned.stdout

    for name, value in workspace.variables.items():
        monkeypatch.setenv(name, value)
    st.cache_resource.clear()
    app = AppTest.from_file(str(REPOSITORY / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert [(m.label, m.value) for m in app.metric][2] == ("Chunks", "22")
    app.text_input(key="search_question").set_value("Can I work remotely?")
    app = app.button[0].click().run()
    assert any("remote_work_policy.md" in element.value for element in app.text)
    st.cache_resource.clear()


def test_rag_through_the_cli(workspace):
    if not has_ollama():
        pytest.skip("BLOCKED, not passed: Ollama or the model is unavailable")
    assert workspace.run("scripts/ingest.py").returncode == 0
    answered = workspace.run(
        "scripts/ask.py", "How many days of annual leave do employees get?", "--json"
    )
    assert answered.returncode == 0, answered.stderr
    response = json.loads(answered.stdout)
    assert response["status"] == "answered" and "25" in response["answer"]

    small = workspace.run("scripts/ask.py", "Anything", OLLAMA_NUM_CTX="700")
    assert small.returncode == 2
    assert "prompt_too_large" in small.stdout


def test_corrupt_file_fails_alone(workspace):
    (workspace.documents / "broken.pdf").write_bytes(b"%PDF-1.7 garbage")
    done = workspace.run("scripts/ingest.py")
    assert done.returncode == 1
    assert "[corrupt_pdf]" in done.stdout
    assert "12 synced" in done.stdout


@pytest.mark.parametrize(
    ("script", "arguments", "variables", "message"),
    [
        ("scripts/search.py", ["x"], {"CHUNK_SIZE": "abc"}, "Invalid configuration"),
        ("scripts/ingest.py", [], {"CHUNK_OVERLAP": "999"}, "Invalid configuration"),
        (
            "scripts/ingest.py",
            [],
            {"QDRANT_URL": "http://127.0.0.1:1", "QDRANT_RETRIES": "1"},
            "unreachable",
        ),
        ("scripts/search.py", ["x"], {}, "does not exist"),
        ("scripts/inspect_index.py", [], {}, "does not exist"),
        ("scripts/ingest.py", ["--dry-run"], {"MODEL_ASSETS_PATH": "/none.json"}, "not prepared"),
        ("scripts/inspect_embeddings.py", ["x"], {"MODEL_ASSETS_PATH": "/none.json"}, "prepared"),
    ],
)
def test_failures_are_clean_through_the_cli(workspace, script, arguments, variables, message):
    done = workspace.run(script, *arguments, **variables)
    assert done.returncode == 2
    assert message in done.stderr
    assert "Traceback" not in done.stderr


def test_dimension_mismatch_changes_nothing(workspace, client):
    client.create_collection(
        workspace.collection,
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
    )
    done = workspace.run("scripts/ingest.py")
    assert done.returncode == 2
    assert "128 dimensions" in done.stderr
    assert client.count(workspace.collection).count == 0


def test_changed_chunking_is_refused(workspace):
    assert workspace.run("scripts/ingest.py").returncode == 0
    done = workspace.run("scripts/search.py", "x", CHUNK_SIZE="128", CHUNK_OVERLAP="24")
    assert done.returncode == 2
    assert "another model or processing configuration" in done.stderr


@pytest.mark.parametrize(
    "script",
    [
        "ingest",
        "prepare_assets",
        "inspect_embeddings",
        "inspect_index",
        "search",
        "ask",
        "evaluate",
        "benchmark",
    ],
)
def test_every_script_has_help(script):
    done = command(f"scripts/{script}.py", "--help")
    assert done.returncode == 0
    assert "usage:" in done.stdout
