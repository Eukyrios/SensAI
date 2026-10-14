# Tests for the model bake-off script: speed computation, missing models and the results table.

from scripts import benchmark as bake_off
from sensai.llm.ollama import ChatStats, OllamaError
from tests.fakes import Returns


class BenchmarkClient:
    def __init__(self, installed=("qwen3.5:9b",), stats=ChatStats(eval_count=50, eval_duration_ns=2_000_000_000)):
        self.installed = installed
        self.stats = stats
        self.prompts = []

    def require_model(self, model):
        if model not in self.installed:
            raise OllamaError("model_missing", model)

    def chat_stream(self, model, messages, **options):
        self.prompts.append(messages[-1]["content"])
        yield "こんにちは"
        return self.stats


def test_tokens_per_second_uses_the_generation_time():
    stats = ChatStats(eval_count=50, eval_duration_ns=2_000_000_000, total_duration_ns=10_000_000_000)
    assert bake_off.tokens_per_second(stats) == 25.0


def test_tokens_per_second_falls_back_to_the_total_time():
    assert bake_off.tokens_per_second(ChatStats(eval_count=10, total_duration_ns=5_000_000_000)) == 2.0


def test_tokens_per_second_without_usable_stats():
    assert bake_off.tokens_per_second(None) is None
    assert bake_off.tokens_per_second(ChatStats(eval_count=10)) is None


def test_print_reply_prints_text_and_returns_the_stats(capsys):
    stats = bake_off.print_reply(BenchmarkClient().chat_stream("m", [{"content": "x"}]))
    assert stats.eval_count == 50 and "こんにちは" in capsys.readouterr().out


def test_benchmark_model_averages_every_prompt():
    client = BenchmarkClient()
    row = bake_off.benchmark_model(client, "qwen3.5:9b")
    assert row == ["qwen3.5:9b", "25.00 t/s", "not measured"]
    assert len(client.prompts) == len(bake_off.TEST_CASES)
    assert all(prompt.endswith(bake_off.CONCISE_INSTRUCTION) for prompt in client.prompts)


def test_missing_model_is_skipped(capsys):
    assert bake_off.benchmark_model(BenchmarkClient(installed=()), "gemma4:12b-it-qat") is None
    assert "ollama pull gemma4:12b-it-qat" in capsys.readouterr().out


def test_main_prints_a_table_of_the_installed_models(monkeypatch, capsys):
    monkeypatch.setattr(bake_off, "OllamaClient", Returns(BenchmarkClient()))
    bake_off.main()
    out = capsys.readouterr().out
    assert "BAKE-OFF RESULTS" in out and "qwen3.5:9b" in out and "25.00 t/s" in out
    assert "qwen3.5:4b" not in out.split("BAKE-OFF RESULTS")[1]     # not installed: not in the table
