"""Tests for the CLI renderer and slash commands."""
import asyncio
from unittest.mock import MagicMock

import pytest

from sensai.core.events import Error, GuardrailTriggered, Token, TurnFinished
from sensai.surfaces.cli import CLIRenderer, run_command


@pytest.mark.asyncio
async def test_cli_renderer_renders_tokens_and_turn_finished(capsys):
    renderer = CLIRenderer()

    async def stream_tokens():
        yield Token(turn_id="t1", run_id="r1", text="Konnichiwa ")
        yield Token(turn_id="t1", run_id="r1", text="world!")
        yield TurnFinished(turn_id="t1", model="qwen3.5:9b")

    await renderer.render_stream(stream_tokens())
    captured = capsys.readouterr().out
    assert captured == "Konnichiwa world!\n"


@pytest.mark.asyncio
async def test_cli_renderer_handles_guardrail_triggered(capsys):
    renderer = CLIRenderer()

    async def stream_guardrail():
        yield GuardrailTriggered(
            turn_id="t1",
            side="input",
            rule="no_pii",
            action="block",
        )

    await renderer.render_stream(stream_guardrail())
    captured = capsys.readouterr().err
    assert "[guardrail] block: no_pii" in captured


@pytest.mark.asyncio
async def test_cli_renderer_handles_all_error_kinds(capsys):
    renderer = CLIRenderer()

    async def stream_errors():
        yield Error(turn_id="t1", kind="model_missing", message="ollama pull qwen3.5:9b")
        yield Error(turn_id="t1", kind="connection", message="Connection failed")
        yield Error(turn_id="t1", kind="timeout", message="Request timed out")
        yield Error(turn_id="t1", kind="bad_request", message="Bad request parameters")

    await renderer.render_stream(stream_errors())
    captured = capsys.readouterr().err

    assert "Model not found" in captured
    assert "ollama pull qwen3.5:9b" in captured
    assert "Cannot reach Ollama" in captured
    assert "Request timed out" in captured
    assert "Bad request parameters" in captured


@pytest.mark.asyncio
async def test_cli_renderer_handles_cancellation(capsys):
    renderer = CLIRenderer()

    async def cancelling_stream():
        yield Token(turn_id="t1", run_id="r1", text="Hello")
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await renderer.render_stream(cancelling_stream())

    captured = capsys.readouterr()
    assert "Hello" in captured.out
    assert "[interrupted]" in captured.err


def test_slash_command_help(capsys):
    session = MagicMock()
    assert run_command(session, "/help") is True
    captured = capsys.readouterr().out
    assert "/help" in captured
    assert "/model" in captured


def test_slash_command_quit():
    session = MagicMock()
    assert run_command(session, "/quit") is False


def test_slash_command_reset(capsys):
    session = MagicMock()
    assert run_command(session, "/reset") is True
    session.reset.assert_called_once()
    assert "History cleared." in capsys.readouterr().out


def test_slash_command_model_show(capsys):
    session = MagicMock()
    session.spec.model = "qwen3.5:9b"
    session.client.list_models.return_value = ["qwen3.5:9b", "qwen3.5:4b"]

    assert run_command(session, "/model") is True
    captured = capsys.readouterr().out
    assert "Model: qwen3.5:9b" in captured
    assert "Installed: qwen3.5:9b, qwen3.5:4b" in captured


def test_slash_command_model_switch(capsys):
    session = MagicMock()
    assert run_command(session, "/model qwen3.5:4b") is True
    session.set_model.assert_called_once_with("qwen3.5:4b")
    assert "Model: qwen3.5:4b" in capsys.readouterr().out


def test_slash_command_unknown(capsys):
    session = MagicMock()
    assert run_command(session, "/unknown") is True
    assert "Unknown command /unknown" in capsys.readouterr().out