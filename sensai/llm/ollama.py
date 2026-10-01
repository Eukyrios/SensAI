"""Direct HTTP calls to Ollama. One client per server; each call says which model and options to use."""
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
        self._show_cache: dict[str, dict] = {}

    def chat_stream(self, model: str, messages: list[Message], *, num_ctx: int,
                    temperature: float = 0.3, think: bool = False) -> Iterator[str | Thought]:
        """Stream the reply token by token. num_ctx is required: never rely on Ollama's default."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        r = self._post("/api/chat", payload, stream=True)
        if r.status_code == 404:
            r.close()
            raise OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")
        if r.status_code == 400:
            text = r.text
            r.close()
            raise OllamaError("bad_request", f"Ollama rejected the request: {text}")
        if not r.ok:
            text = r.text
            r.close()
            raise OllamaError("stream", f"Ollama error {r.status_code}: {text}")

        try:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    raise OllamaError("stream", "malformed json")

                if "error" in chunk:
                    raise OllamaError("stream", chunk["error"])

                msg = chunk.get("message", {})
                
                thinking = msg.get("thinking", "")
                if thinking:
                    yield Thought(thinking)

                token = msg.get("content", "")
                if token:
                    yield token

                if chunk.get("done"):
                    return ChatStats.from_chunk(chunk)
        except (requests.exceptions.ChunkedEncodingError, requests.ConnectionError):
            raise OllamaError("connection", "Connection broken in the middle of the reply")
        except requests.exceptions.ReadTimeout:
            raise OllamaError("timeout", "Timeout in the middle of the reply")
        finally:
            r.close()

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

    def embed(self, text: str, model: str) -> list[float]:
        r = self._post("/api/embed", {"model": model, "input": text})
        if not r.ok:
            raise OllamaError("bad_request", f"Embedding error {r.status_code}: {r.text}")
        return r.json()["embeddings"][0]

    def _get_show(self, model: str) -> dict:
        if model in self._show_cache:
            return self._show_cache[model]
        r = self._post("/api/show", {"model": model})
        if r.status_code == 404:
            raise OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")
        if not r.ok:
            raise OllamaError("bad_request", f"Cannot fetch model details ({r.status_code}): {r.text}")
        data = r.json()
        self._show_cache[model] = data
        return data

    def capabilities(self, model: str) -> set[str]:
        data = self._get_show(model)
        return set(data.get("capabilities", []))

    def context_length(self, model: str) -> int | None:
        data = self._get_show(model)
        model_info = data.get("model_info", {})
        arch = model_info.get("general.architecture")
        if not arch:
            return None
        return model_info.get(f"{arch}.context_length")

    def _post(self, path: str, payload: dict, stream: bool = False):
        return self._request("POST", path, json=payload, stream=stream)

    def _request(self, method: str, path: str, **kwargs):
        try:
            return requests.request(method, f"{self.url}{path}", timeout=self.timeout_s, **kwargs)
        except requests.ConnectionError:
            raise OllamaError("connection", f"Cannot reach Ollama at {self.url} (is `ollama serve` running?)") from None
        except requests.Timeout:
            raise OllamaError("timeout", "Ollama did not answer in time") from None