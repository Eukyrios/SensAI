"""Shared test doubles."""
from sensai.core.config import load_config


def make_config(**sections):
    """Real merged config plus per-test section overrides."""
    config = load_config()
    for section, values in sections.items():
        config.setdefault(section, {}).update(values)
    return config


class FakeClient:
    """Stands in for OllamaClient: streams fixed tokens or raises."""

    def __init__(self, tokens=("Kon", "nichiwa"), error=None):
        self.tokens = tokens
        self.error = error
        self.calls = []

    def chat_stream(self, spec, messages):
        self.calls.append((spec, messages))
        if self.error:
            raise self.error
        yield from self.tokens
