# Shared test doubles: a fake Ollama client, a config builder, and small stubs for monkeypatch.

from sensai.core.config import load_config
from sensai.llm.ollama import OllamaError


def make_config(**sections):
    return load_config(env={}, cli=sections or None)


class FakeClient:

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


class ScriptedClient(FakeClient):

    def __init__(self, *replies, stats=None):
        super().__init__(stats=stats)
        self.replies = list(replies)

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, messages, options))
        reply = self.replies.pop(0)
        yield from (reply[i:i + 7] for i in range(0, len(reply), 7))
        return self.stats


class Returns:

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value


class Recorder:

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)


class AsyncRecorder(Recorder):

    async def __call__(self, *args, **kwargs):
        self.calls.append(args)


class ScriptedInput:

    def __init__(self, items):
        self.items = iter(items)

    def __call__(self, prompt=""):
        item = next(self.items, EOFError)
        if isinstance(item, BaseException) or (isinstance(item, type) and issubclass(item, BaseException)):
            raise item
        return item


async def events_stream(*events):
    for event in events:
        yield event


async def events_then_raise(events, error):
    for event in events:
        yield event
    raise error


def always_true():
    return True
