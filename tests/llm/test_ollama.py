import json

import pytest
import requests

from sensai import ollama_client
from sensai.ollama_client import OllamaClient, OllamaError


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
    return OllamaClient("http://ollama:11434/", "m")


def patch_post(monkeypatch, result):
    def fake_post(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(ollama_client.requests, "post", fake_post)


def test_url_trailing_slash_removed(client):
    assert client.url == "http://ollama:11434"


def test_chat_stream_yields_tokens(monkeypatch, client):
    lines = [{"message": {"content": "a"}}, b"", {"message": {"content": ""}},
             {"message": {"content": "b"}, "done": True}, {"message": {"content": "c"}}]
    patch_post(monkeypatch, FakeResponse(lines=lines))
    assert list(client.chat_stream([])) == ["a", "b"]


@pytest.mark.parametrize("exc, msg", [
    (requests.ConnectionError(), "injoignable"),
    (requests.Timeout(), "timeout"),
])
def test_chat_stream_network_errors(monkeypatch, client, exc, msg):
    patch_post(monkeypatch, exc)
    with pytest.raises(OllamaError, match=msg):
        list(client.chat_stream([]))


def test_chat_stream_model_not_found(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(status=404))
    with pytest.raises(OllamaError, match="introuvable"):
        list(client.chat_stream([]))


def test_chat_stream_http_error(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(status=500, text="boom"))
    with pytest.raises(OllamaError, match="500"):
        list(client.chat_stream([]))


def test_chat_stream_error_chunk(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(lines=[{"error": "bad"}]))
    with pytest.raises(OllamaError, match="bad"):
        list(client.chat_stream([]))


def test_embed_ok(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(payload={"embeddings": [[0.1, 0.2]]}))
    assert client.embed("x", "e") == [0.1, 0.2]


def test_embed_connection_error(monkeypatch, client):
    patch_post(monkeypatch, requests.ConnectionError())
    with pytest.raises(OllamaError, match="injoignable"):
        client.embed("x", "e")


def test_embed_http_error(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(status=500, text="boom"))
    with pytest.raises(OllamaError, match="embedding"):
        client.embed("x", "e")
