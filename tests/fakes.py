"""Shared test doubles."""
from sensai.core.config import load_config


def make_config(**sections):
    """Real merged config with no environment overrides, plus per-test section overrides."""
    return load_config(env={}, cli=sections or None)


class FakeClient:
    """Stands in for OllamaClient: streams fixed tokens or raises."""

    def __init__(self, tokens=("Kon", "nichiwa"), error=None, startup_error=None):
        self.tokens = tokens
        self.error = error
        self.startup_error = startup_error
        self.calls = []

    def require_model(self, model):
        if self.startup_error:
            raise self.startup_error

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, messages, options))
        if self.error:
            raise self.error
        yield from self.tokens
