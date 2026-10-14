# Tests for the CLI: renderer, slash commands, live input line and startup.

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
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("SENSAI_LOGGING__PATH", str(path))
    return path


@pytest.fixture(autouse=True)
def installed_models(monkeypatch):
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


def test_usage_line_without_ttft():
    line = cli.usage_line(TurnFinished(turn_id="t", model="m", eval_count=3, total_duration_ns=2_000_000_000))
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
    await cli.render_turn(session, "hello", cli.CLIRenderer(cli.PlainOutput(), statuses.append))
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


from tests.conftest import memory_repository


def test_list_sessions(monkeypatch, capsys):
    repo = memory_repository()
    sid = repo.create_session("qwen3.5:9b", "Lesson 1")
    monkeypatch.setattr(cli, "open_repository", lambda config: repo)
    run_cli_test(monkeypatch, [], argv=["--list-sessions"])
    out = capsys.readouterr().out
    assert sid in out and "qwen3.5:9b" in out and "Lesson 1" in out


def test_list_sessions_empty(monkeypatch, capsys):
    run_cli_test(monkeypatch, [], argv=["--list-sessions"])
    assert "No stored sessions." in capsys.readouterr().out


def test_resume_session(monkeypatch, capsys):
    repo = memory_repository()
    stored = Session(make_config(), client=FakeClient(), repo=repo)
    stored.record("user", "earlier")
    stored.record("assistant", "Konnichiwa")
    monkeypatch.setattr("sensai.orchestration.session.open_repository", lambda config: repo)
    seen = []
    monkeypatch.setattr(cli, "plain_chat", lambda session, logger: seen.append(session))
    cli.main(["--session", stored.id])
    assert f"Resumed session {stored.id} (2 messages)" in capsys.readouterr().out
    assert seen[0].id == stored.id and [m["content"] for m in seen[0].history] == ["earlier", "Konnichiwa"]


def test_unknown_session_exits_with_error(monkeypatch):
    with pytest.raises(SystemExit, match="session not found: nope") as e:
        cli.main(["--session", "nope"])
    assert e.value.code not in (0, None)


import contextlib
import threading

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from sensai.core.config import CONFIG_PATH
from sensai.llm.ollama import OllamaError


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


async def live(monkeypatch, session, script):
    monkeypatch.setattr(cli, "patch_stdout", lambda raw=False: contextlib.nullcontext())
    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        chat = asyncio.create_task(cli.live_chat(session))
        for step in script:
            if isinstance(step, str):
                keys.send_text(step)
            await asyncio.sleep(step if isinstance(step, (int, float)) else 0.05)
        await asyncio.wait_for(chat, 5)


def live_session(client):
    from tests.fakes import make_config
    return Session(make_config(), client=client)


class SlowClient(FakeClient):

    def __init__(self, delay=0.1, count=5):
        super().__init__()
        self.delay, self.count, self.asked = delay, count, []

    def chat_stream(self, model, messages, **options):
        self.asked.append(messages[-1]["content"])
        for i in range(self.count):
            threading.Event().wait(self.delay)
            yield f"t{i} "


def test_bad_agent(monkeypatch, tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[agents]\ndefault = "nobody"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown agent"):
        run(monkeypatch, [], argv=["--config", str(p)])


async def test_blank_lines_are_ignored(monkeypatch, capsys):
    client = FakeClient()
    await live(monkeypatch, live_session(client), ["   \r", "\x04"])
    assert client.calls == []


def test_ctrl_c_at_prompt(monkeypatch, capsys):
    run(monkeypatch, [KeyboardInterrupt], argv=["--config", str(CONFIG_PATH)])
    assert "Sensai" in capsys.readouterr().out


async def test_ctrl_c_does_nothing_during_a_reply(monkeypatch, capsys):
    client = SlowClient(delay=0.1, count=5)
    session = live_session(client)
    await live(monkeypatch, session, ["first\r", 0.2, "\x03", 1.0, "/quit\r"])
    out = capsys.readouterr().out
    assert "[interrupted]" not in out and "t4" in out
    assert session.history[-1]["content"].startswith("t0 ")


async def test_ctrl_c_while_idle_does_nothing(monkeypatch):
    client = FakeClient()
    await live(monkeypatch, live_session(client), ["\x03", 0.2, "hello\r", 0.3, "\x04"])
    assert len(client.calls) == 1


async def test_ctrl_d_during_a_reply_quits_without_running_queued_questions(monkeypatch, capsys):
    client = SlowClient(delay=0.3, count=10)
    await live(monkeypatch, live_session(client), ["first\r", 0.4, "second\r", 0.3, "\x04"])
    assert capsys.readouterr().err.count("[interrupted]") == 1
    assert client.asked == ["first"]


async def test_escape_stops_the_reply_and_keeps_the_chat(monkeypatch, capsys):
    client = SlowClient(delay=0.3, count=10)
    session = live_session(client)
    await live(monkeypatch, session, ["first\r", 1.0, "\x1b", 0.5, "/quit\r"])
    captured = capsys.readouterr()
    assert captured.err.count("[interrupted]") == 1 and "t9" not in captured.out
    question, partial = session.history
    assert question == {"role": "user", "content": "first"}
    assert partial["role"] == "assistant" and partial["content"].startswith("t0 ")


async def test_escape_while_idle_does_nothing(monkeypatch):
    client = FakeClient()
    await live(monkeypatch, live_session(client), ["\x1b", 0.3, "\x04"])
    assert client.calls == []


def test_full_session(monkeypatch, capsys):
    run(monkeypatch, ["", "hello", "/reset", "/quit"], client=FakeClient())
    out = capsys.readouterr().out
    assert "Konnichiwa" in out and "History cleared." in out


def test_interrupt_during_reply(monkeypatch, capsys):
    run(monkeypatch, ["hello"], client=FakeClient(error=KeyboardInterrupt()))
    assert capsys.readouterr().err.count("[interrupted]") == 1


async def test_live_chat_answers_then_quits(monkeypatch, capsys):
    await live(monkeypatch, live_session(FakeClient(stats=STATS)), ["hello\r", 0.3, "/quit\r"])
    out = capsys.readouterr().out
    assert "> hello\nKonnichiwa\n" in out and "102 prompt + 11 reply tokens" in out


def test_main_uses_the_live_chat_in_a_terminal(monkeypatch):
    used = []

    async def fake_live(session, logger):
        used.append(session)
    monkeypatch.setattr(cli, "live_chat", fake_live)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "Session", lambda config: Session(config, client=FakeClient()))
    cli.main([])
    assert len(used) == 1


def test_missing_config(monkeypatch):
    with pytest.raises(SystemExit, match="config not found"):
        run(monkeypatch, [], argv=["--config", "/nope.toml"])


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


def test_module_entrypoint(monkeypatch):
    import runpy
    monkeypatch.setattr("sys.argv", ["sensai"])
    monkeypatch.setattr("builtins.input", lambda p="": (_ for _ in ()).throw(EOFError))
    runpy.run_module("sensai", run_name="__main__")


def test_ollama_error_printed(monkeypatch, capsys):
    run(monkeypatch, ["hello"], client=FakeClient(error=OllamaError("connection", "down")))
    err = capsys.readouterr().err
    assert "[error] Cannot reach Ollama" in err and "ollama serve" in err


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
    assert asked == []


async def test_type_ahead_is_queued_and_runs_after_the_reply(monkeypatch, capsys):
    client = SlowClient()
    await live(monkeypatch, live_session(client),
               ["first\r", 0.15, "second\r", 1.5, "/quit\r"])
    out = capsys.readouterr().out
    assert client.asked == ["first", "second"]
    assert out.index("> first") < out.index("t4") < out.index("> second")


def test_unknown_command_not_sent_to_model(monkeypatch, capsys):
    client = FakeClient()
    run(monkeypatch, ["/nope"], client=client)
    assert "Unknown command /nope. Type /help" in capsys.readouterr().out
    assert client.calls == []


async def test_renderer_prints_structured_output_as_json(capsys):
    async def stream():
        yield ev.StructuredOutput(turn_id="t", run_id="r", data={"correction": "学校に行きます"})
        yield ev.StructuredOutput(turn_id="t", run_id="r", data={"ok": True}, attempts=2)
    await CLIRenderer().render_stream(stream())
    out = capsys.readouterr().out
    assert '"correction": "学校に行きます"' in out
    assert out.count("repaired") == 1


def test_agent_flag_selects_the_agent(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "plain_chat", lambda session, logger: seen.append(session.spec))
    monkeypatch.setattr(cli, "Session", lambda config: Session(config, client=FakeClient()))
    cli.main(["--agent", "checker"])
    assert seen[0].name == "checker" and seen[0].output_schema["required"][0] == "sentence"


def test_unknown_agent_flag_exits(monkeypatch):
    with pytest.raises(SystemExit, match="unknown agent 'nobody'"):
        cli.main(["--agent", "nobody"])


def test_metrics_and_trace_commands_show_the_last_turn(monkeypatch, capsys):
    run(monkeypatch, ["/metrics", "/trace", "hello", "/metrics", "/trace"], client=FakeClient(stats=STATS))
    out = capsys.readouterr().out
    before, after = out.split("Konnichiwa", 1)
    assert "No metrics recorded yet." in before and "No trace recorded yet." in before
    assert "No metrics recorded yet." not in after and "No trace recorded yet." not in after
    assert "sensei" in after
