"""Tests for the CLI renderer and slash commands."""
import asyncio
import json
from unittest.mock import MagicMock

import pytest

from sensai.core import events as ev
from sensai.core.events import Error, GuardrailTriggered, Token, TurnFinished
from sensai.llm.ollama import ChatStats, OllamaClient, Thought
from sensai.orchestration.session import Session
from sensai.surfaces import cli
from sensai.surfaces.cli import CLIRenderer, run_command
from tests.fakes import FakeClient, make_config


def run_cli_test(monkeypatch, inputs: list[str], client=None, argv: list[str] | None = None) -> None:
    """Helper running main() by mocking input sequence and client."""
    client = client or FakeClient()
    monkeypatch.setattr(cli, "Session", lambda config: Session(config, client=client))
    iter_inputs = iter(inputs)
    
    def fake_input(prompt=""):
        try:
            return next(iter_inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    try:
        cli.main(argv or [])
    except SystemExit:
        pass


@pytest.fixture(autouse=True)
def event_log(monkeypatch, tmp_path):
    """Tests never write into the repository's logs/ folder."""
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("SENSAI_LOGGING__PATH", str(path))
    return path


@pytest.fixture(autouse=True)
def installed_models(monkeypatch):
    """Tests never reach a real Ollama: pretend every model is installed."""
    monkeypatch.setattr(OllamaClient, "require_model", lambda self, model: None)


@pytest.mark.asyncio
async def test_cli_renderer_renders_tokens_and_turn_finished(capsys):
    renderer = CLIRenderer()

    async def stream_tokens():
        yield Token(turn_id="t1", run_id="r1", text="Konnichiwa ")
        yield Token(turn_id="t1", run_id="r1", text="world!")
        yield TurnFinished(turn_id="t1", model="qwen3.5:9b")

    await renderer.render_stream(stream_tokens())
    captured = capsys.readouterr().out
    assert "Konnichiwa world!" in captured


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
    assert "[guardrail] input block: no_pii" in captured


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


def test_model_flag_and_eof(monkeypatch, capsys):
    run_cli_test(monkeypatch, [], argv=["--model", "qwen3.5:4b"])
    out = capsys.readouterr().out
    assert "Sensai (qwen3.5:4b) — type /help for commands" in out


def test_help_lists_every_command(monkeypatch, capsys):
    client = FakeClient()
    run_cli_test(monkeypatch, ["/help"], client=client)
    out = capsys.readouterr().out
    for usage, text in cli.COMMANDS.values():
        assert usage in out and text in out


STATS = ChatStats(prompt_eval_count=102, eval_count=11, total_duration_ns=1_200_000_000)


def test_usage_line_after_reply(monkeypatch, capsys):
    run_cli_test(monkeypatch, ["hello"], client=FakeClient(stats=STATS))
    out = capsys.readouterr().out
    assert "Konnichiwa" in out
    assert "102 prompt + 11 reply tokens" in out and "1.2s" in out


def test_no_usage_line_without_stats(monkeypatch, capsys):
    run_cli_test(monkeypatch, ["hello"], client=FakeClient())
    assert "reply tokens" not in capsys.readouterr().out


def test_usage_without_ttft():
    line = cli.usage(TurnFinished(turn_id="t", model="m", eval_count=3, total_duration_ns=2_000_000_000))
    assert line == "0 prompt + 3 reply tokens · 2.0s"


def test_plain_output_ends_lines_only_when_needed(capsys):
    out = cli.PlainOutput()
    out.end_line()
    out.write("Kon")
    out.write("nichiwa")
    out.end_line()
    out.write("done\n")
    out.end_line()
    assert capsys.readouterr().out == "Konnichiwa\ndone\n"


def test_live_output_prints_finished_lines_and_keeps_the_partial_one(capsys):
    changes = []
    out = cli.LiveOutput()
    out.on_change = lambda: changes.append(out.partial)
    out.write("Red: 赤 (aka)\nBlue: ")
    out.write("青")
    assert capsys.readouterr().out == "Red: 赤 (aka)\n"
    assert out.partial == "Blue: 青" and changes == ["Blue: ", "Blue: 青"]
    out.end_line()
    assert capsys.readouterr().out == "Blue: 青\n" and out.partial == ""
    out.end_line()
    assert capsys.readouterr().out == ""


async def test_render_turn_reports_status():
    statuses = []
    session = Session(make_config(), client=FakeClient(stats=STATS))
    await cli.render_turn(session, "hello", cli.PlainOutput(), statuses.append)
    assert statuses[0] == "thinking…" and "replying…" in statuses and statuses[-1] is None


def test_thinking_is_shown_before_the_answer(monkeypatch, capsys):
    client = FakeClient(tokens=(Thought("Is it\na greeting?"), "Kon", "nichiwa"))
    run_cli_test(monkeypatch, ["hello"], client=client)
    out = capsys.readouterr().out
    assert "thinking:\nIs it\na greeting?\n\nKonnichiwa" in out


def test_no_thinking_header_without_reasoning(monkeypatch, capsys):
    run_cli_test(monkeypatch, ["hello"], client=FakeClient())
    assert "thinking:" not in capsys.readouterr().out


def test_think_command(monkeypatch, capsys):
    client = FakeClient()
    run_cli_test(monkeypatch, ["/think", "/think on", "hello", "/think off"], client=client)
    out = capsys.readouterr().out
    assert "Thinking: off" in out and "Thinking: on" in out


def test_think_on_rejected_for_a_model_that_cannot_reason(monkeypatch, capsys):
    run_cli_test(monkeypatch, ["/think on"], client=FakeClient(thinkers=()))
    assert "can't reason" in capsys.readouterr().err


def test_model_switch_reports_thinking_turned_off(monkeypatch, capsys):
    client = FakeClient(thinkers=("qwen3.5:9b",))
    run_cli_test(monkeypatch, ["/think on", "/model qwen3.5:4b"], client=client)
    assert "Thinking turned off: qwen3.5:4b" in capsys.readouterr().out


def test_dim_lines_dims_each_line(monkeypatch):
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli.dim_lines("a\n\nb") == "\033[2ma\033[0m\n\n\033[2mb\033[0m"


async def test_renderer_shows_guardrail_events(capsys):
    async def stream():
        yield ev.GuardrailTriggered(turn_id="t", side="input", rule="pii", action="redact")
    await cli.CLIRenderer().render_stream(stream())
    assert "[guardrail] input redact: pii" in capsys.readouterr().err


def test_events_are_logged_to_the_configured_path(monkeypatch, event_log):
    run_cli_test(monkeypatch, ["hello"], client=FakeClient(stats=STATS))
    records = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    kinds = [r["event_type"] for r in records]
    assert kinds[0] == "TurnStarted" and kinds[-1] == "TurnFinished" and "Token" in kinds
    assert len({r["turn_id"] for r in records}) == 1