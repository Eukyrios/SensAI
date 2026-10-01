"""Shared test doubles."""
from sensai.core.config import load_config
from sensai.llm.ollama import OllamaError


def make_config(**sections):
    """Real merged config with no environment overrides, plus per-test section overrides."""
    return load_config(env={}, cli=sections or None)


class FakeClient:
    """Stands in for OllamaClient: streams fixed tokens or raises."""

    def __init__(self, tokens=("Kon", "nichiwa"), error=None, startup_error=None,
                 models=("qwen3.5:4b", "qwen3.5:9b"), stats=None, thinkers=("qwen3.5:4b", "qwen3.5:9b")):
        self.tokens = tokens
        self.thinkers = set(thinkers)
        self.stats = stats
        self.error = error
        self.startup_error = startup_error
        self.models = list(models)
        self.calls = []

    def capabilities(self, model):
        return {"completion", "thinking"} if model in self.thinkers else {"completion"}

    def list_models(self):
        return self.models

    def require_model(self, model):
        if self.startup_error:
            raise self.startup_error
        if model not in self.models:
            raise OllamaError("model_missing", f"Model '{model}' not found (run `ollama pull {model}`)")

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, messages, options))
        if self.error:
            raise self.error
        yield from self.tokens
        return self.stats
