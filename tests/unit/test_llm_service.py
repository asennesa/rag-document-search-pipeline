"""Uses the real ollama client over httpx's mock transport, so wire format and error
handling are exercised for real, without a server."""

import hashlib
import json

import httpx
import ollama
import pytest

from rag_pipeline.errors import GenerationError
from rag_pipeline.rag.llm_service import GeneratedAnswer, LLMService

TEMPLATE = "template text"
GOOD = {"answer": "25 days.", "citations": ["S1"], "insufficient_evidence": False}


class Server:
    def __init__(self, show=None, chat=None):
        self.show = show or (lambda: httpx.Response(200, json=show_body()))
        self.chat = chat or (lambda: chat_response(GOOD))
        self.requests = []

    def __call__(self, request):
        self.requests.append((request.url.path, json.loads(request.content or b"{}")))
        return self.show() if request.url.path == "/api/show" else self.chat()

    def paths(self):
        return [path for path, _ in self.requests]


def show_body(template=TEMPLATE, context=32768):
    return {"template": template, "model_info": {"qwen3.context_length": context}}


def chat_response(content, done_reason="stop"):
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(
        200,
        json={
            "model": "m",
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": done_reason,
            "prompt_eval_count": 120,
            "eval_count": 30,
        },
    )


def make_service(server, num_ctx=4096):
    client = ollama.Client(host="http://ollama.test", transport=httpx.MockTransport(server))
    return LLMService(
        client,
        model="qwen3:test",
        num_ctx=num_ctx,
        max_output_tokens=512,
        temperature=0,
        template_sha256=hashlib.sha256(TEMPLATE.encode()).hexdigest(),
    )


def category(server, **kwargs):
    with pytest.raises(GenerationError) as caught:
        make_service(server, **kwargs).generate("sys", "user")
    return caught.value


def test_request_uses_chat_messages_schema_and_options():
    server = Server()
    make_service(server).generate("system text", "user text")
    _, body = server.requests[-1]
    assert body["model"] == "qwen3:test"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert body["format"] == GeneratedAnswer.model_json_schema()
    assert body["options"]["num_ctx"] == 4096
    assert body["options"]["num_predict"] == 512
    assert body["options"]["temperature"] == 0
    assert not body.get("tools")
    assert body["think"] is False


def test_result_is_parsed_with_token_counts_and_latency():
    result = make_service(Server()).generate("s", "u")
    assert result.answer == "25 days."
    assert result.citations == ("S1",)
    assert result.insufficient_evidence is False
    assert (result.prompt_tokens, result.output_tokens) == (120, 30)
    assert result.latency_ms >= 0


def test_verification_runs_once():
    server = Server()
    service = make_service(server)
    service.generate("s", "u")
    service.generate("s", "u")
    assert server.paths() == ["/api/show", "/api/chat", "/api/chat"]


def test_missing_model_is_distinct_and_never_pulled():
    server = Server(show=lambda: httpx.Response(404, json={"error": "model not found"}))
    error = category(server)
    assert error.category == "model_missing"
    assert "ollama pull qwen3:test" in str(error)
    assert "/api/pull" not in server.paths()


def test_unknown_template_is_refused_before_generation():
    server = Server(show=lambda: httpx.Response(200, json=show_body("changed")))
    assert category(server).category == "model_unverified"
    assert "/api/chat" not in server.paths()


def test_context_above_the_model_limit_is_refused():
    server = Server(show=lambda: httpx.Response(200, json=show_body(context=2048)))
    assert category(server).category == "model_unverified"


def test_connection_refused_is_distinct():
    def refuse():
        raise httpx.ConnectError("refused")

    error = category(Server(show=refuse))
    assert error.category == "ollama_unavailable"
    assert "ollama serve" in str(error)


def test_timeout_is_distinct():
    def slow():
        raise httpx.ReadTimeout("slow")

    assert category(Server(chat=slow)).category == "timeout"


def test_server_error_is_reported_without_details():
    server = Server(chat=lambda: httpx.Response(500, json={"error": "private detail"}))
    error = category(server)
    assert error.category == "ollama_unavailable"
    assert "private" not in str(error)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"answer": "x", "citations": []}',
        '{"answer": "x", "insufficient_evidence": false}',
        '{"answer": 3, "citations": [], "insufficient_evidence": false}',
        "",
    ],
)
def test_malformed_content_is_distinct(content):
    assert category(Server(chat=lambda: chat_response(content))).category == "malformed_response"


def test_empty_answer_is_malformed():
    body = {"answer": "  ", "citations": [], "insufficient_evidence": False}
    assert category(Server(chat=lambda: chat_response(body))).category == "malformed_response"


def test_output_limit_is_incomplete_even_with_cut_off_json():
    server = Server(chat=lambda: chat_response('{"answer": "cut of', "length"))
    assert category(server).category == "incomplete"
