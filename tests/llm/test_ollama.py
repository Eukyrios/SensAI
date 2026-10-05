import json

import pytest
import requests

from sensai.llm import ollama
from sensai.llm.ollama import ChatStats, OllamaClient, OllamaError, Thought

OPTS = {"num_ctx": 2048, "temperature": 0.1}


class FakeResponse:
    def __init__(self, status=200, lines=(), text="", payload=None, fail_with=None):
        self.status_code = status
        self.ok = status < 400
        self.text = text
        self._lines = lines
        self._payload = payload
        self._fail_with = fail_with
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            yield line if isinstance(line, bytes) else json.dumps(line).encode()
        if self._fail_with:
            raise self._fail_with

    def close(self):
        self.closed = True

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


@pytest.mark.parametrize("exc, kind", [
    (requests.exceptions.ChunkedEncodingError(), "connection"),
    (requests.ConnectionError(), "connection"),
    (requests.exceptions.ReadTimeout(), "timeout"),
])
def test_network_failure_mid_stream(monkeypatch, client, exc, kind):
    response = FakeResponse(lines=[{"message": {"content": "a"}}], fail_with=exc)
    patch_post(monkeypatch, response)
    stream = client.chat_stream("m", [], **OPTS)
    assert next(stream) == "a"
    with pytest.raises(OllamaError, match="middle of the reply") as e:
        next(stream)
    assert e.value.kind == kind
    assert response.closed


def test_malformed_line_mid_stream(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(lines=[b"not json"]))
    with pytest.raises(OllamaError, match="malformed") as e:
        list(client.chat_stream("m", [], **OPTS))
    assert e.value.kind == "stream"


def test_response_closed_when_consumer_stops_early(monkeypatch, client):
    response = FakeResponse(lines=[{"message": {"content": "a"}}, {"message": {"content": "b"}}])
    patch_post(monkeypatch, response)
    stream = client.chat_stream("m", [], **OPTS)
    assert next(stream) == "a"
    stream.close()
    assert response.closed


def test_response_closed_after_normal_end(monkeypatch, client):
    response = FakeResponse(lines=[{"message": {"content": "a"}, "done": True}])
    patch_post(monkeypatch, response)
    assert list(client.chat_stream("m", [], **OPTS)) == ["a"]
    assert response.closed


def drain(stream):
    """Collect the tokens and the generator's return value (the ChatStats)."""
    tokens = []
    while True:
        try:
            tokens.append(next(stream))
        except StopIteration as stop:
            return tokens, stop.value


FINAL = {"message": {"content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 35, "eval_count": 12, "total_duration": 535582993,
         "load_duration": 158906296, "prompt_eval_duration": 323145881, "eval_duration": 49855632}


def test_final_chunk_stats_are_returned(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(lines=[{"message": {"content": "こんにちは"}}, FINAL]))
    tokens, stats = drain(client.chat_stream("m", [], **OPTS))
    assert tokens == ["こんにちは"]
    assert stats == ChatStats(prompt_eval_count=35, eval_count=12, total_duration_ns=535582993,
                              load_duration_ns=158906296, prompt_eval_duration_ns=323145881,
                              eval_duration_ns=49855632, done_reason="stop")


def test_plain_loop_sees_only_text(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(lines=[{"message": {"content": "a"}}, FINAL]))
    assert list(client.chat_stream("m", [], **OPTS)) == ["a"]


def test_no_final_chunk_returns_none(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(lines=[{"message": {"content": "a"}}]))
    assert drain(client.chat_stream("m", [], **OPTS)) == (["a"], None)


def test_missing_stat_fields_default_to_zero():
    assert ChatStats.from_chunk({"done": True}) == ChatStats()


SHOW = {"capabilities": ["completion", "tools", "thinking"],
        "model_info": {"general.architecture": "qwen35", "qwen35.context_length": 262144}}


def test_capabilities_and_context_length(monkeypatch, client):
    sent = patch_post(monkeypatch, FakeResponse(payload=SHOW))
    assert client.capabilities("qwen3.5:9b") == {"completion", "tools", "thinking"}
    assert client.context_length("qwen3.5:9b") == 262144
    assert (sent["method"], sent["url"], sent["json"]) == ("POST", "http://ollama:11434/api/show",
                                                           {"model": "qwen3.5:9b"})


def test_show_is_cached_per_model(monkeypatch, client):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(kwargs["json"]["model"])
        return FakeResponse(payload=SHOW)
    monkeypatch.setattr(ollama.requests, "request", fake_request)
    client.capabilities("a")
    client.context_length("a")
    client.capabilities("b")
    assert calls == ["a", "b"]


def test_missing_details_are_empty(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(payload={}))
    assert client.capabilities("old") == set()
    assert client.context_length("old") is None


@pytest.mark.parametrize("status, kind, match", [(404, "model_missing", "ollama pull"),
                                                 (500, "bad_request", "model details")])
def test_show_errors(monkeypatch, client, status, kind, match):
    patch_post(monkeypatch, FakeResponse(status=status, text="boom"))
    with pytest.raises(OllamaError, match=match) as e:
        client.capabilities("x")
    assert e.value.kind == kind


def test_thinking_comes_as_thoughts_before_the_answer(monkeypatch, client):
    lines = [{"message": {"thinking": "Is 7 ", "content": ""}},
             {"message": {"thinking": "prime?", "content": ""}},
             {"message": {"content": "Yes"}}, FINAL]
    sent = patch_post(monkeypatch, FakeResponse(lines=lines))
    tokens, stats = drain(client.chat_stream("m", [], think=True, **OPTS))
    assert tokens == [Thought("Is 7 "), Thought("prime?"), "Yes"]
    assert sent["json"]["think"] is True and stats.eval_count == 12


def test_no_thinking_field_means_text_only(monkeypatch, client):
    patch_post(monkeypatch, FakeResponse(lines=[{"message": {"content": "a", "thinking": ""}}, FINAL]))
    assert list(client.chat_stream("m", [], **OPTS)) == ["a"]


@pytest.mark.parametrize("fmt", ["json", {"type": "object"}])
def test_format_is_sent_when_given(monkeypatch, client, fmt):
    sent = patch_post(monkeypatch, FakeResponse(lines=[FINAL]))
    list(client.chat_stream("m", [], format=fmt, **OPTS))
    assert sent["json"]["format"] == fmt


def test_no_format_by_default(monkeypatch, client):
    sent = patch_post(monkeypatch, FakeResponse(lines=[FINAL]))
    list(client.chat_stream("m", [], **OPTS))
    assert "format" not in sent["json"]
