# Tests for the model benchmark: answer checking, speed, the sensei's settings and the saved results.

import json

from scripts import benchmark
from sensai.core.agent_spec import AgentSpec
from sensai.llm.ollama import ChatStats, OllamaError
from tests.fakes import Returns

SPEC = AgentSpec(name="sensei", model="placeholder", num_ctx=8192, system_prompt="You teach Japanese.")
STATS = ChatStats(eval_count=50, eval_duration_ns=2_000_000_000, load_duration_ns=3_000_000_000)


class BenchmarkClient:

    def __init__(self, installed=("qwen3.5:9b",), reply="食べませんでした (tabemasen deshita)"):
        self.installed = installed
        self.reply = reply
        self.requests = []

    def require_model(self, model):
        if model not in self.installed:
            raise OllamaError("model_missing", model)

    def chat_stream(self, model, messages, **options):
        self.requests.append((model, messages, options))
        yield self.reply
        return STATS


def test_contains_expected_ignores_case():
    assert benchmark.contains_expected("It marks the TOPIC.", ("topic",))
    assert not benchmark.contains_expected("It marks the subject.", ("topic",))


def test_every_case_has_an_expected_answer():
    assert len(benchmark.TEST_CASES) == 20
    assert all(case.expected for case in benchmark.TEST_CASES)


def test_tokens_per_second_uses_the_generation_time():
    assert benchmark.tokens_per_second(STATS) == 25.0
    assert benchmark.tokens_per_second(ChatStats(eval_count=10, total_duration_ns=5_000_000_000)) == 2.0
    assert benchmark.tokens_per_second(None) is None
    assert benchmark.tokens_per_second(ChatStats(eval_count=10)) is None


def test_cases_use_the_sensei_settings():
    client = BenchmarkClient()
    benchmark.benchmark_model(client, SPEC, "qwen3.5:9b")
    model, messages, options = client.requests[0]
    assert model == "qwen3.5:9b" and options["num_ctx"] == 8192
    assert messages[0] == {"role": "system", "content": "You teach Japanese."}
    assert messages[1]["content"].endswith(benchmark.CONCISE_INSTRUCTION)


def test_model_result_scores_accuracy_speed_and_load():
    result = benchmark.benchmark_model(BenchmarkClient(), SPEC, "qwen3.5:9b")
    assert len(result.cases) == 20
    assert 0 < result.accuracy < 1                      # the fixed reply only answers the conjugation case
    assert result.cases[2].correct                       # 食べませんでした
    assert result.average_speed == 25.0 and result.load_s == 3.0
    assert result.table_row()[0] == "qwen3.5:9b"


def test_missing_model_is_skipped(capsys):
    assert benchmark.benchmark_model(BenchmarkClient(installed=()), SPEC, "gemma4:12b-it-qat") is None
    assert "ollama pull gemma4:12b-it-qat" in capsys.readouterr().out


def test_results_are_saved_as_json(tmp_path):
    result = benchmark.benchmark_model(BenchmarkClient(), SPEC, "qwen3.5:9b")
    path = tmp_path / "results.json"
    benchmark.save_results([result], SPEC, path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["settings"]["num_ctx"] == 8192 and report["settings"]["questions"] == 20
    model = report["models"][0]
    assert model["model"] == "qwen3.5:9b" and len(model["cases"]) == 20
    assert model["cases"][2]["correct"] is True


def test_main_runs_the_installed_models(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(benchmark, "OllamaClient", Returns(BenchmarkClient()))
    monkeypatch.setattr(benchmark, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setattr(benchmark, "save_results", Returns(None))
    benchmark.main()
    out = capsys.readouterr().out
    assert "qwen3.5:9b" in out and "Skipped" in out and "Accuracy" in out
