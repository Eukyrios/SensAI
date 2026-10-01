"""Direct HTTP calls to Ollama. One client per server; each call says which model and options to use."""
import json
from collections.abc import Generator, Iterator
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
    """A piece of the model's reasoning, sent before the answer when thinking is on."""
    text: str


@dataclass(frozen=True)
class ChatStats:
    """Telemetry from Ollama's final chunk ("done": true). Durations are in nanoseconds."""
    prompt_eval_count: int = 0          # tokens in the prompt
    eval_count: int = 0                 # tokens generated
    total_duration_ns: int = 0
    load_duration_ns: int = 0           # time spent loading the model
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    done_reason: str | None = None      # "stop", "length", ...

    @classmethod
    def from_chunk(cls, chunk: dict) -> "ChatStats":
        return cls(
            prompt_eval_count=chunk.get("prompt_eval_count", 0),
            eval_count=chunk.get("eval_count", 0),
            total_duration_ns=chunk.get("total_duration", 0),
            load_duration_ns=chunk.get("load_duration", 0),
            prompt_eval_duration_ns=chunk.get("prompt_eval_duration", 0),
            eval_duration_ns=chunk.get("eval_duration", 0),
            done_reason=chunk.get("done_reason"),
        )


class OllamaClient:
    def __init__(self, url: str, timeout_s: int = 60):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self._shown: dict[str, dict] = {}       # /api/show results per model

    def chat_stream(self, model: str, messages: list[Message], *, num_ctx: int,
                    temperature: float = 0.3, think: bool = False
                    ) -> Generator[str | Thought, None, ChatStats | None]:
        """Stream the reply token by token. num_ctx is required: never rely on Ollama's default.

        Yields the answer as plain str tokens. When the model reasons first (think=True), its
        reasoning comes as Thought items before the answer, never mixed into the text.
        The return value is the ChatStats of the final chunk (None if the stream ended without one):
        `stats = yield from client.chat_stream(...)`, or StopIteration.value when iterating by hand.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        r = self._post("/api/chat", payload, stream=True)
        if r.status_code == 404:
            raise OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")
        if r.status_code == 400:
            raise OllamaError("bad_request", f"Ollama rejected the request: {r.text}")
        if not r.ok:
            raise OllamaError("stream", f"Ollama error {r.status_code}: {r.text}")

        try:
            for chunk in _chunks(r):
                if "error" in chunk:        # can arrive mid-stream while HTTP is still 200
                    raise OllamaError("stream", chunk["error"])
                message = chunk.get("message", {})
                if message.get("thinking"):
                    yield Thought(message["thinking"])
                if message.get("content"):
                    yield message["content"]
                if chunk.get("done"):
                    return ChatStats.from_chunk(chunk)
        finally:
            r.close()                       # also on early exit: frees the connection, stops Ollama
        return None

    def list_models(self) -> list[str]:
        """Names of the installed models, e.g. ["llama3.2:latest"] (GET /api/tags)."""
        r = self._request("GET", "/api/tags")
        if not r.ok:
            raise OllamaError("bad_request", f"Cannot list models ({r.status_code}): {r.text}")
        return [m["name"] for m in r.json().get("models", [])]

    def require_model(self, model: str) -> None:
        """Raise model_missing unless the model is installed. A name without a tag means ':latest'."""
        wanted = model if ":" in model else f"{model}:latest"
        if wanted not in self.list_models():
            raise OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")

    def show(self, model: str) -> dict:
        """Model details from POST /api/show, cached per model (they only change on `ollama pull`)."""
        if model not in self._shown:
            r = self._post("/api/show", {"model": model})
            if r.status_code == 404:
                raise OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")
            if not r.ok:
                raise OllamaError("bad_request", f"Cannot read model details ({r.status_code}): {r.text}")
            self._shown[model] = r.json()
        return self._shown[model]

    def capabilities(self, model: str) -> set[str]:
        """What the model supports, e.g. {"completion", "tools", "thinking", "vision"}."""
        return set(self.show(model).get("capabilities") or [])

    def context_length(self, model: str) -> int | None:
        """The model's maximum context in tokens (an upper bound for num_ctx), if Ollama reports it."""
        for key, value in self.show(model).get("model_info", {}).items():
            if key.endswith(".context_length"):
                return value
        return None

    def embed(self, text: str, model: str) -> list[float]:
        r = self._post("/api/embed", {"model": model, "input": text})
        if not r.ok:
            raise OllamaError("bad_request", f"Embedding error {r.status_code}: {r.text}")
        return r.json()["embeddings"][0]

    def _post(self, path: str, payload: dict, stream: bool = False):
        return self._request("POST", path, json=payload, stream=stream)

    def _request(self, method: str, path: str, **kwargs):
        try:
            return requests.request(method, f"{self.url}{path}", timeout=self.timeout_s, **kwargs)
        except requests.ConnectionError:
            raise OllamaError("connection", f"Cannot reach Ollama at {self.url} (is `ollama serve` running?)") from None
        except requests.Timeout:
            raise OllamaError("timeout", "Ollama did not answer in time") from None


def _chunks(r) -> Iterator[dict]:
    """Parse the NDJSON stream, turning failures in the middle of it into OllamaError."""
    try:
        for line in r.iter_lines():
            if line:
                yield json.loads(line)
    except requests.Timeout:
        raise OllamaError("timeout", "Ollama stopped answering in the middle of the reply") from None
    except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
        raise OllamaError("connection", "Lost the connection to Ollama in the middle of the reply") from None
    except json.JSONDecodeError:
        raise OllamaError("stream", "Ollama sent a malformed line in the reply") from None
