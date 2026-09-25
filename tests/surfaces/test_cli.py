import sys

import pytest

from sensai import main as cli
from sensai.ollama_client import OllamaError
from tests.conftest import FakeClient


def run(monkeypatch, inputs, argv=(), client=None):
    it = iter(inputs)

    def fake_input(prompt=""):
        item = next(it, EOFError)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys, "argv", ["sensai", *argv])
    if client is not None:
        monkeypatch.setattr(cli.Agent, "__init__", _init_with(client))
    cli.main()


def _init_with(client):
    orig = cli.Agent.__init__

    def init(self, config, model=None):
        orig(self, config, model)
        self.client = client
    return init


def test_full_session(monkeypatch, capsys):
    run(monkeypatch, ["", "/persona", "/persona casual", "salut", "/reset", "/quit"],
        client=FakeClient())
    out = capsys.readouterr().out
    assert "Persona actuel : formal" in out
    assert "Persona : casual" in out
    assert "Bonjour" in out
    assert "Historique effacé." in out


def test_eof_exits(monkeypatch, capsys):
    run(monkeypatch, [], argv=["--model", "x"])
    assert "Sensai (x)" in capsys.readouterr().out


def test_ollama_error_printed(monkeypatch, capsys):
    run(monkeypatch, ["salut"], client=FakeClient(error=OllamaError("down")))
    assert "[erreur] down" in capsys.readouterr().err


def test_keyboard_interrupt_during_reply(monkeypatch, capsys):
    run(monkeypatch, ["salut"], client=FakeClient(error=KeyboardInterrupt()))
    assert "[interrompu]" in capsys.readouterr().out


def test_missing_config(monkeypatch):
    with pytest.raises(SystemExit, match="config introuvable"):
        run(monkeypatch, [], argv=["--config", "/nope.yaml"])


def test_custom_config(monkeypatch, capsys):
    from sensai.config import CONFIG_PATH
    run(monkeypatch, [KeyboardInterrupt], argv=["--config", str(CONFIG_PATH)])
    assert "Sensai" in capsys.readouterr().out
