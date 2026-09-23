"""Appels HTTP directs à Ollama — propriétaire : A"""
import json
from typing import Iterator
import requests

from .interfaces import Message


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, url: str, model: str, think: bool = False,
                 temperature: float = 0.3, timeout: int = 60):
        self.url = url.rstrip("/")
        self.model = model
        self.think = think
        self.temperature = temperature
        self.timeout = timeout

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "options": {"temperature": self.temperature},
        }
        try:
            r = requests.post(f"{self.url}/api/chat", json=payload,
                              stream=True, timeout=self.timeout)
        except requests.ConnectionError:
            raise OllamaError(f"Ollama injoignable sur {self.url} (lance `ollama serve`)")
        except requests.Timeout:
            raise OllamaError("Ollama ne répond pas (timeout)")

        if r.status_code == 404:
            raise OllamaError(f"Modèle '{self.model}' introuvable (`ollama pull {self.model}`)")
        if not r.ok:
            raise OllamaError(f"Erreur Ollama {r.status_code}: {r.text}")

        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise OllamaError(chunk["error"])
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break

    def embed(self, text: str, model: str) -> list[float]:
        """Utilisé par le RAG (B)."""
        try:
            r = requests.post(f"{self.url}/api/embed",
                              json={"model": model, "input": text},
                              timeout=self.timeout)
        except requests.ConnectionError:
            raise OllamaError(f"Ollama injoignable sur {self.url}")
        if not r.ok:
            raise OllamaError(f"Erreur embedding {r.status_code}: {r.text}")
        return r.json()["embeddings"][0]