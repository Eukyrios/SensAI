# Ollama HTTP transport: chat streaming, telemetry, embeddings, /api/tags and /api/show.

import json
from collections.abc import Iterator
from dataclasses import dataclass

import requests

from ..core.errors import ErrorKind
from ..core.messages import Message


class OllamaError(RuntimeError):
    def __init__(self, kind: ErrorKind, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Thought:
    text: str


@dataclass(frozen=True)
class ChatStats:
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    done_reason: str = ""

    @classmethod
    def from_chunk(cls, chunk: dict) -> "ChatStats":
        return cls(
            prompt_eval_count=chunk.get("prompt_eval_count", 0),
            eval_count=chunk.get("eval_count", 0),
            total_duration_ns=chunk.get("total_duration", 0),
            load_duration_ns=chunk.get("load_duration", 0),
            prompt_eval_duration_ns=chunk.get("prompt_eval_duration", 0),
            eval_duration_ns=chunk.get("eval_duration", 0),
            done_reason=chunk.get("done_reason", "")
        )


class OllamaClient:
    def __init__(self, url: str, timeout_s: int = 60):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self._details_by_model: dict[str, dict] = {}

    def chat_stream(self, model: str, messages: list[Message], *, num_ctx: int,
                    temperature: float = 0.3, think: bool = False,
                    format: dict | str | None = None) -> Iterator[str | Thought]:
        payload = _chat_payload(model, messages, num_ctx, temperature, think, format)
        response = self._post("/api/chat", payload, stream=True)
        try:
            _raise_for_chat_status(response, model)
            return (yield from _read_reply(response))
        finally:
            response.close()

    def list_models(self) -> list[str]:
        response = self._request("GET", "/api/tags")
        if not response.ok:
            raise OllamaError("bad_request", f"Cannot list models ({response.status_code}): {response.text}")
        return [entry["name"] for entry in response.json().get("models", [])]

    def require_model(self, model: str) -> None:
        full_name = model if ":" in model else f"{model}:latest"
        if full_name not in self.list_models():
            raise _model_missing(model)

    def embed(self, text: str, model: str) -> list[float]:
        response = self._post("/api/embed", {"model": model, "input": text})
        if not response.ok:
            raise OllamaError("bad_request", f"Embedding error {response.status_code}: {response.text}")
        return response.json()["embeddings"][0]

    def capabilities(self, model: str) -> set[str]:
        return set(self._model_details(model).get("capabilities", []))

    def context_length(self, model: str) -> int | None:
        model_info = self._model_details(model).get("model_info", {})
        architecture = model_info.get("general.architecture")
        if not architecture:
            return None
        return model_info.get(f"{architecture}.context_length")

    def _model_details(self, model: str) -> dict:
        if model not in self._details_by_model:
            self._details_by_model[model] = self._fetch_model_details(model)
        return self._details_by_model[model]

    def _fetch_model_details(self, model: str) -> dict:
        response = self._post("/api/show", {"model": model})
        if response.status_code == 404:
            raise _model_missing(model)
        if not response.ok:
            raise OllamaError("bad_request",
                              f"Cannot fetch model details ({response.status_code}): {response.text}")
        return response.json()

    def _post(self, path: str, payload: dict, stream: bool = False):
        return self._request("POST", path, json=payload, stream=stream)

    def _request(self, method: str, path: str, **kwargs):
        try:
            return requests.request(method, f"{self.url}{path}", timeout=self.timeout_s, **kwargs)
        except requests.ConnectionError:
            raise OllamaError("connection", f"Cannot reach Ollama at {self.url} (is `ollama serve` running?)") from None
        except requests.Timeout:
            raise OllamaError("timeout", "Ollama did not answer in time") from None


def _chat_payload(model: str, messages: list[Message], num_ctx: int, temperature: float,
                  think: bool, format: dict | str | None) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if format is not None:
        payload["format"] = format
    return payload


def _raise_for_chat_status(response, model: str) -> None:
    if response.status_code == 404:
        raise _model_missing(model)
    if response.status_code == 400:
        raise OllamaError("bad_request", f"Ollama rejected the request: {response.text}")
    if not response.ok:
        raise OllamaError("stream", f"Ollama error {response.status_code}: {response.text}")


def _read_reply(response) -> Iterator[str | Thought]:
    for chunk in _stream_chunks(response):
        if "error" in chunk:
            raise OllamaError("stream", chunk["error"])
        yield from _chunk_pieces(chunk.get("message", {}))
        if chunk.get("done"):
            return ChatStats.from_chunk(chunk)
    return None


def _stream_chunks(response) -> Iterator[dict]:
    try:
        for line in response.iter_lines():
            if line:
                yield _parse_chunk(line)
    except (requests.exceptions.ChunkedEncodingError, requests.ConnectionError):
        raise OllamaError("connection", "Connection broken in the middle of the reply")
    except requests.exceptions.ReadTimeout:
        raise OllamaError("timeout", "Timeout in the middle of the reply")


def _parse_chunk(line: bytes) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        raise OllamaError("stream", "malformed json") from None


def _chunk_pieces(message: dict) -> Iterator[str | Thought]:
    if message.get("thinking"):
        yield Thought(message["thinking"])
    if message.get("content"):
        yield message["content"]


def _model_missing(model: str) -> OllamaError:
    return OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")
