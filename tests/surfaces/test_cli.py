import pytest

from sensai.core.config import CONFIG_PATH
from sensai.llm.ollama import OllamaClient, OllamaError
from sensai.orchestration.session import Session
from sensai.surfaces import cli
from tests.fakes import FakeClient


@pytest.fixture(autouse=True)
def installed_models(monkeypatch):
    """Tests never reach a real Ollama: pretend every model is installed."""
    monkeypatch.setattr(OllamaClient, "require_model", lambda self, model: None)


def run(monkeypatch, inputs, argv=(), client=None):
    it = iter(inputs)

    def fake_input(prompt=""):
        item = next(it, EOFError)
        if isinstance(item, BaseException) or (isinstance(item, type) and issubclass(item, BaseException)):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)
    if client is not None:
        monkeypatch.setattr(cli, "Session", lambda config: Session(config, client=client))
    cli.main(list(argv))


def test_full_session(monkeypatch, capsys):
    run(monkeypatch, ["", "hello", "/reset", "/quit"], client=FakeClient())
    out = capsys.readouterr().out
    assert "Konnichiwa" in out and "History cleared." in out


def test_model_flag_and_eof(monkeypatch, capsys):
    run(monkeypatch, [], argv=["--model", "x"])
    assert "Sensai (x)" in capsys.readouterr().out


def test_ollama_error_printed(monkeypatch, capsys):
    run(monkeypatch, ["hello"], client=FakeClient(error=OllamaError("connection", "down")))
    assert "[error] down" in capsys.readouterr().err


def test_interrupt_during_reply(monkeypatch, capsys):
    run(monkeypatch, ["hello"], client=FakeClient(error=KeyboardInterrupt()))
    assert "[interrupted]" in capsys.readouterr().out


def test_missing_config(monkeypatch):
    with pytest.raises(SystemExit, match="config not found"):
        run(monkeypatch, [], argv=["--config", "/nope.toml"])


@pytest.mark.parametrize("error, shown", [
    (OllamaError("model_missing", "Model 'x' not found (run `ollama pull x`)"), "ollama pull x"),
    (OllamaError("connection", "Cannot reach Ollama"), "Cannot reach Ollama"),
])
def test_startup_check_stops_before_prompt(monkeypatch, error, shown):
    asked = []
    monkeypatch.setattr("builtins.input", lambda p="": asked.append(p))
    monkeypatch.setattr(cli, "Session", lambda config: Session(config, client=FakeClient(startup_error=error)))
    with pytest.raises(SystemExit, match=shown):
        cli.main([])
    assert asked == []          # the prompt never appeared


def test_bad_agent(monkeypatch, tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[agents]\ndefault = "nobody"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown agent"):
        run(monkeypatch, [], argv=["--config", str(p)])


def test_ctrl_c_at_prompt(monkeypatch, capsys):
    run(monkeypatch, [KeyboardInterrupt], argv=["--config", str(CONFIG_PATH)])
    assert "Sensai" in capsys.readouterr().out


def test_module_entrypoint(monkeypatch):
    import runpy
    monkeypatch.setattr("sys.argv", ["sensai"])
    monkeypatch.setattr("builtins.input", lambda p="": (_ for _ in ()).throw(EOFError))
    runpy.run_module("sensai", run_name="__main__")


def test_model_command(monkeypatch, capsys):
    client = FakeClient()
    run(monkeypatch, ["/model", "/model qwen3.5:4b", "hello", "/model nope"], client=client)
    out, err = capsys.readouterr()
    assert "Model: qwen3.5:9b" in out and "Installed: qwen3.5:4b, qwen3.5:9b" in out
    assert "Model: qwen3.5:4b (history kept)" in out
    assert client.calls[-1][0] == "qwen3.5:4b"
    assert "ollama pull nope" in err


def test_model_command_lists_none(monkeypatch, capsys):
    client = FakeClient()
    client.list_models = lambda: []
    run(monkeypatch, ["/model"], client=client)
    assert "Installed: none" in capsys.readouterr().out


def test_unknown_command_not_sent_to_model(monkeypatch, capsys):
    client = FakeClient()
    run(monkeypatch, ["/nope"], client=client)
    assert "Unknown command /nope. Type /help" in capsys.readouterr().out
    assert client.calls == []


def test_help_lists_every_command(monkeypatch, capsys):
    client = FakeClient()
    run(monkeypatch, ["/help"], client=client)
    out = capsys.readouterr().out
    for usage, text in cli.COMMANDS.values():
        assert usage in out and text in out
    assert client.calls == []
