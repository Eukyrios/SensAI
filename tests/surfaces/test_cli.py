import pytest

from sensai.core.config import CONFIG_PATH
from sensai.llm.ollama import ChatStats, OllamaClient, OllamaError
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
    out = capsys.readouterr().out
    assert "Sensai (x) — type /help for commands" in out and "/reset" not in out


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


STATS = ChatStats(prompt_eval_count=102, eval_count=11, total_duration_ns=1_200_000_000)


def test_usage_line_after_reply(monkeypatch, capsys):
    run(monkeypatch, ["hello"], client=FakeClient(stats=STATS))
    out = capsys.readouterr().out
    assert "Konnichiwa" in out
    assert "102 prompt + 11 reply tokens" in out and "1.2s" in out and "first token" in out


def test_no_usage_line_without_stats(monkeypatch, capsys):
    run(monkeypatch, ["hello"], client=FakeClient())
    assert "reply tokens" not in capsys.readouterr().out


def test_usage_without_ttft():
    from sensai.core.events import TurnFinished
    line = cli.usage(TurnFinished(turn_id="t", model="m", eval_count=3, total_duration_ns=2_000_000_000))
    assert line == "0 prompt + 3 reply tokens · 2.0s"



# --- output modes -------------------------------------------------------------------------------

def test_plain_output_ends_lines_only_when_needed(capsys):
    out = cli.PlainOutput()
    out.end_line()                       # nothing written yet: no blank line
    out.write("Kon")
    out.write("nichiwa")
    out.end_line()
    out.write("done\n")
    out.end_line()                       # already at a line start
    assert capsys.readouterr().out == "Konnichiwa\ndone\n"


def test_live_output_prints_finished_lines_and_keeps_the_partial_one(capsys):
    changes = []
    out = cli.LiveOutput()
    out.on_change = lambda: changes.append(out.partial)
    out.write("Red: 赤 (aka)\nBlue: ")
    out.write("青")
    assert capsys.readouterr().out == "Red: 赤 (aka)\n"     # only the finished line is printed
    assert out.partial == "Blue: 青" and changes == ["Blue: ", "Blue: 青"]
    out.end_line()
    assert capsys.readouterr().out == "Blue: 青\n" and out.partial == ""
    out.end_line()                                           # nothing pending: prints nothing
    assert capsys.readouterr().out == ""


async def test_render_turn_reports_status():
    from sensai.orchestration.session import Session as S
    from tests.fakes import make_config
    statuses = []
    session = S(make_config(), client=FakeClient(stats=STATS))
    await cli.render_turn(session, "hello", cli.PlainOutput(), statuses.append)
    assert statuses[0] == "thinking…" and "replying…" in statuses and statuses[-1] is None


# --- live input line ----------------------------------------------------------------------------

import asyncio
import contextlib
import threading

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


async def live(monkeypatch, session, script):
    """Run live_chat with keys fed from `script`: a list of strings or seconds to wait."""
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


async def test_live_chat_answers_then_quits(monkeypatch, capsys):
    await live(monkeypatch, live_session(FakeClient(stats=STATS)), ["hello\r", 0.3, "/quit\r"])
    out = capsys.readouterr().out
    assert "> hello\nKonnichiwa\n" in out and "102 prompt + 11 reply tokens" in out


class SlowClient(FakeClient):
    """Streams one token every `delay` seconds and records each message it was asked."""

    def __init__(self, delay=0.1, count=5):
        super().__init__()
        self.delay, self.count, self.asked = delay, count, []

    def chat_stream(self, model, messages, **options):
        self.asked.append(messages[-1]["content"])
        for i in range(self.count):
            threading.Event().wait(self.delay)
            yield f"t{i} "


async def test_type_ahead_is_queued_and_runs_after_the_reply(monkeypatch, capsys):
    client = SlowClient()
    await live(monkeypatch, live_session(client),
               ["first\r", 0.15, "second\r", 1.5, "/quit\r"])     # "second" typed mid-reply
    out = capsys.readouterr().out
    assert client.asked == ["first", "second"]
    assert out.index("> first") < out.index("t4") < out.index("> second")   # never mid-reply


async def test_escape_stops_the_reply_and_keeps_the_chat(monkeypatch, capsys):
    client = SlowClient(delay=0.3, count=10)
    session = live_session(client)
    await live(monkeypatch, session, ["first\r", 1.0, "\x1b", 0.5, "/quit\r"])   # Esc mid-reply
    out = capsys.readouterr().out
    assert "[interrupted]" in out and "t9" not in out
    question, partial = session.history                  # the chat went on, history kept
    assert question == {"role": "user", "content": "first"}
    assert partial["role"] == "assistant" and partial["content"].startswith("t0 ")


async def test_ctrl_c_does_nothing_during_a_reply(monkeypatch, capsys):
    client = SlowClient(delay=0.1, count=5)
    session = live_session(client)
    await live(monkeypatch, session, ["first\r", 0.2, "\x03", 1.0, "/quit\r"])
    out = capsys.readouterr().out
    assert "[interrupted]" not in out and "t4" in out        # the reply finished normally
    assert session.history[-1]["content"].startswith("t0 ")


async def test_ctrl_d_during_a_reply_quits_without_running_queued_questions(monkeypatch, capsys):
    client = SlowClient(delay=0.3, count=10)
    await live(monkeypatch, live_session(client), ["first\r", 0.4, "second\r", 0.3, "\x04"])
    assert "[interrupted]" in capsys.readouterr().out
    assert client.asked == ["first"]                      # "second" was queued but never sent


async def test_escape_while_idle_does_nothing(monkeypatch):
    client = FakeClient()
    await live(monkeypatch, live_session(client), ["\x1b", 0.3, "\x04"])
    assert client.calls == []


async def test_ctrl_c_while_idle_does_nothing(monkeypatch):
    client = FakeClient()
    await live(monkeypatch, live_session(client), ["\x03", 0.2, "hello\r", 0.3, "\x04"])
    assert len(client.calls) == 1                         # still running: "hello" was answered


async def test_blank_lines_are_ignored(monkeypatch, capsys):
    client = FakeClient()
    await live(monkeypatch, live_session(client), ["   \r", "\x04"])
    assert client.calls == []


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


# --- thinking -----------------------------------------------------------------------------------

from sensai.llm.ollama import Thought


def test_thinking_is_shown_before_the_answer(monkeypatch, capsys):
    client = FakeClient(tokens=(Thought("Is it\na greeting?"), "Kon", "nichiwa"))
    run(monkeypatch, ["hello"], client=client)
    out = capsys.readouterr().out
    assert "thinking:\nIs it\na greeting?\n\nKonnichiwa" in out


def test_no_thinking_header_without_reasoning(monkeypatch, capsys):
    run(monkeypatch, ["hello"], client=FakeClient())
    assert "thinking:" not in capsys.readouterr().out


def test_think_command(monkeypatch, capsys):
    client = FakeClient()
    run(monkeypatch, ["/think", "/think on", "hello", "/think off", "/think maybe"], client=client)
    out = capsys.readouterr().out
    assert out.count("Thinking: off") == 2 and "Thinking: on" in out
    assert client.calls[0][2]["think"] is True                 # the request asked for reasoning
    assert "Usage: /think [on|off]" in out


def test_think_on_rejected_for_a_model_that_cannot_reason(monkeypatch, capsys):
    run(monkeypatch, ["/think on"], client=FakeClient(thinkers=()))
    assert "can't reason" in capsys.readouterr().err


def test_model_switch_reports_thinking_turned_off(monkeypatch, capsys):
    client = FakeClient(thinkers=("qwen3.5:9b",))
    run(monkeypatch, ["/think on", "/model qwen3.5:4b"], client=client)
    assert "Thinking turned off: qwen3.5:4b" in capsys.readouterr().out


def test_dim_lines_dims_each_line(monkeypatch):
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli.dim_lines("a\n\nb") == "\033[2ma\033[0m\n\n\033[2mb\033[0m"
