import json

import pytest
import requests

from sensai.llm import ollama
from sensai.llm.ollama import OllamaClient, OllamaError

OPTS = {"num_ctx": 2048, "temperature": 0.1}


class FakeResponse:
    def __init__(self, status=200, lines=(), text="", payload=None):
        self.status_code = status
        self.ok = status < 400
        self.text = text
        self._lines = lines
        self._payload = payload

    def iter_lines(self):
        for line in self._lines:
            yield line if isinstance(line, bytes) else json.dumps(line).encode()

    def json(self):
        return self._payload


@pytest.fixture
def client():
    return OllamaClient("http://ollama:11434/")


def patch_post(monkeypatch, result):
    sent = {}

    def fake_request(method, url, **kwargs):
        sent.update(method=method, url=url, **kwargs)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(ollama.requests, "request", fake_request)
    return sent


def test_chat_stream_sends_options_and_yields_tokens(monkeypatch, client):
    lines = [{"message": {"content": "a"}}, b"", {"message": {"content": ""}},
             {"message": {"content": "b"}, "done": True}, {"message": {"content": "c"}}]
    sent = patch_post(monkeypatch, FakeResponse(lines=lines))
    assert list(client.chat_stream("m", [], **OPTS)) == ["a", "b"]
    assert sent["url"] == "http://ollama:11434/api/chat"
    assert sent["json"]["model"] == "m"
    assert sent["json"]["options"] == {"temperature": 0.1, "num_ctx": 2048}


@pytest.mark.parametrize("exc, kind", [
    (requests.ConnectionError(), "connection"),
    (requests.Timeout(), "timeout"),
])
def test_network_errors(monkeypatch, client, exc, kind):
    patch_post(monkeypatch, exc)
    with pytest.raises(OllamaError) as e:
        list(client.chat_stream("m", [], **OPTS))
    assert e.value.kind == kind


@pytest.mark.parametrize("status, kind", [(404, "model_missing"), (400, "bad_request"), (500, "stream")])
def test_http_errors(monkeypatch, client, status, kind):
    patch_post(monkeypatch, FakeResponse(status=status, text="boom"))
    with pytest.raises(OllamaError) as e:
        list(client.chat_stream("m", [], **OPTS))
    assert e.value.kind == kind


def test_mid_stream_error(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(lines=[{"message": {"content": "a"}}, {"error": "bad"}]))
    with pytest.raises(OllamaError, match="bad") as e:
        list(client.chat_stream("m", [], **OPTS))
    assert e.value.kind == "stream"


def test_embed(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(payload={"embeddings": [[0.1, 0.2]]}))
    assert client.embed("x", "e") == [0.1, 0.2]


def test_embed_error(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(status=500, text="boom"))
    with pytest.raises(OllamaError, match="Embedding"):
        client.embed("x", "e")


TAGS = {"models": [{"name": "llama3.2:latest"}, {"name": "qwen3.5:4b"}]}


def test_list_models(monkeypatch, client):
    sent = patch_post(monkeypatch, FakeResponse(payload=TAGS))
    assert client.list_models() == ["llama3.2:latest", "qwen3.5:4b"]
    assert (sent["method"], sent["url"]) == ("GET", "http://ollama:11434/api/tags")


def test_list_models_error(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(status=500, text="boom"))
    with pytest.raises(OllamaError, match="Cannot list models"):
        client.list_models()


@pytest.mark.parametrize("name", ["llama3.2", "llama3.2:latest", "qwen3.5:4b"])
def test_require_model_found(monkeypatch, client, name):
    patch_post(monkeypatch, FakeResponse(payload=TAGS))
    client.require_model(name)


@pytest.mark.parametrize("name", ["qwen3.5", "llama3.2:1b", "nope"])
def test_require_model_missing(monkeypatch, client, name):
    patch_post(monkeypatch, FakeResponse(payload=TAGS))
    with pytest.raises(OllamaError, match="ollama pull") as e:
        client.require_model(name)
    assert e.value.kind == "model_missing"


def test_require_model_server_down(monkeypatch, client):
    patch_post(monkeypatch, requests.ConnectionError())
    with pytest.raises(OllamaError) as e:
        client.require_model("llama3.2")
    assert e.value.kind == "connection"
