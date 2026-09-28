"""Direct HTTP calls to Ollama. One client per server; each call says which model and options to use."""
import json
from collections.abc import Iterator

import requests

from ..core.errors import ErrorKind
from ..core.messages import Message


class OllamaError(RuntimeError):
    def __init__(self, kind: ErrorKind, message: str):
        super().__init__(message)
        self.kind = kind


class OllamaClient:
    def __init__(self, url: str, timeout_s: int = 60):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def chat_stream(self, model: str, messages: list[Message], *, num_ctx: int,
                    temperature: float = 0.3, think: bool = False) -> Iterator[str]:
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
            raise OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")
        if r.status_code == 400:
            raise OllamaError("bad_request", f"Ollama rejected the request: {r.text}")
        if not r.ok:
            raise OllamaError("stream", f"Ollama error {r.status_code}: {r.text}")

        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:        # can arrive mid-stream while HTTP is still 200
                raise OllamaError("stream", chunk["error"])
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break

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

    def _post(self, path: str, payload: dict, stream: bool = False):
        return self._request("POST", path, json=payload, stream=stream)

    def _request(self, method: str, path: str, **kwargs):
        try:
            return requests.request(method, f"{self.url}{path}", timeout=self.timeout_s, **kwargs)
        except requests.ConnectionError:
            raise OllamaError("connection", f"Cannot reach Ollama at {self.url} (is `ollama serve` running?)") from None
        except requests.Timeout:
            raise OllamaError("timeout", "Ollama did not answer in time") from None
