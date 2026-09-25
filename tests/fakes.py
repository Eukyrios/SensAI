import pytest

from sensai.config import load_config


@pytest.fixture
def config():
    return load_config()


class FakeClient:
    """Remplace OllamaClient : renvoie des tokens prédéfinis."""

    def __init__(self, tokens=("Bon", "jour"), error=None):
        self.model = "fake"
        self.tokens = tokens
        self.error = error
        self.calls = []

    def chat_stream(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        yield from self.tokens
