"""Unit tests never use the developer's environment, network, or model cache."""

import os
import re
import socket

import pytest

from rag_pipeline.config import Settings


class FakeTokenizer:
    """Words and punctuation as tokens, with spans like a real fast tokenizer."""

    special_token_count = 2

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in re.finditer(r"\w+|[^\w\s]", text)]


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path, request):
    if "unit" not in request.node.path.parts:
        return
    fields = set(Settings.model_fields)
    for name in tuple(os.environ):
        if name.lower() in fields:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("Unit tests must not access network services or download models")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
