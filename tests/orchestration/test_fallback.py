# Tests for the degrade to the sensei: a failed turn is answered by the fallback agent when it can be.

import json
import threading
from dataclasses import replace

from sensai.core import events as ev
from sensai.llm.ollama import OllamaError
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import FakeClient, ScriptedClient, make_config

FINDINGS = {"sentence": "私は学校を行きます", "is_correct": False, "correction": "私は学校に行きます",
            "error_type": "particle", "rule": "Use に for a destination."}
EXPLANATION = "Use に (ni) for where you go."
LOOPING_SPEC = """name = "{name}"
model = "qwen3.5:9b"
num_ctx = 2048
handoffs = ["{target}"]

[output_schema]
type = "object"
required = ["note"]

[output_schema.properties.note]
type = "string"
"""
FALLBACK_SPEC = 'name = "sensei"\nmodel = "qwen3.5:4b"\nnum_ctx = 2048\nsystem_prompt = "Teach."\n'


class FirstCallFails(ScriptedClient):

    def __init__(self, error, *replies):
        super().__init__(*replies)
        self.error = error

    def chat_stream(self, model, messages, **options):
        if not self.calls:
            self.calls.append((model, messages, options))
            raise self.error
        yield from super().chat_stream(model, messages, **options)


class FirstCallStalls(ScriptedClient):

    def __init__(self, *replies):
        super().__init__(*replies)
        self.release = threading.Event()

    def chat_stream(self, model, messages, **options):
        if not self.calls:
            self.calls.append((model, messages, options))
            self.release.wait(5)
        yield from super().chat_stream(model, messages, **options)


def checker_session(client):
    return Session(make_config(agents={"default": "checker"}), client=client)


def write_agents(agents_dir):
    for name, target in (("ping", "pong"), ("pong", "ping")):
        (agents_dir / f"{name}.toml").write_text(LOOPING_SPEC.format(name=name, target=target), encoding="utf-8")
    (agents_dir / "sensei.toml").write_text(FALLBACK_SPEC, encoding="utf-8")


def loop_reply(target):
    return json.dumps({"note": f"to {target}", "handoff": {"to": target, "reason": "loop"}})


async def collect(session, text="私は学校を行きます"):
    return [event async for event in run_turn(session, text)]


def agents_run(events):
    return [event.agent for event in events if isinstance(event, ev.AgentRunStarted)]


async def test_invalid_output_falls_back_to_the_sensei():
    session = checker_session(ScriptedClient("not json", "still not json", EXPLANATION))
    events = await collect(session)
    started = [event for event in events if isinstance(event, ev.AgentRunStarted)]
    assert [run.agent for run in started] == ["checker", "sensei"]
    assert started[1].parent_run_id == started[0].run_id
    assert next(event for event in events if isinstance(event, ev.Error)).kind == "invalid_output"
    assert session.history[-1] == {"role": "assistant", "content": EXPLANATION}


async def test_a_timed_out_agent_falls_back_to_the_sensei():
    session = checker_session(FirstCallStalls(EXPLANATION))
    session.spec = replace(session.spec, timeout_s=0.2)
    events = await collect(session)
    session.client.release.set()
    assert agents_run(events) == ["checker", "sensei"]
    assert [event.kind for event in events if isinstance(event, ev.Error)] == ["timeout"]
    assert events[-2].agent == "sensei" and events[-2].ok is True
    assert session.history[-1]["content"] == EXPLANATION


async def test_a_stopped_loop_falls_back_with_the_last_findings(tmp_path):
    write_agents(tmp_path)
    client = ScriptedClient(loop_reply("pong"), loop_reply("ping"), EXPLANATION)
    session = Session(make_config(agents={"dir": str(tmp_path), "default": "ping"}), client=client)
    events = await collect(session)
    assert agents_run(events) == ["ping", "pong", "sensei"]
    model, messages, _ = client.calls[2]
    assert model == "qwen3.5:4b" and '"to ping"' in messages[1]["content"]
    assert session.history[-1]["content"] == EXPLANATION


async def test_no_fallback_when_ollama_is_down():
    session = checker_session(FirstCallFails(OllamaError("connection", "down"), EXPLANATION))
    events = await collect(session)
    assert agents_run(events) == ["checker"] and session.history == []


async def test_no_second_try_when_the_sensei_itself_failed():
    session = Session(make_config(), client=FakeClient(error=OllamaError("stream", "cut")))
    events = await collect(session, "hello")
    assert agents_run(events) == ["sensei"] and session.history == []


async def test_no_fallback_after_the_turn_budget():
    session = Session(make_config(agents={"default": "checker"}, orchestration={"turn_budget_s": 0.2}),
                      client=FirstCallStalls(EXPLANATION))
    events = await collect(session)
    session.client.release.set()
    assert agents_run(events) == ["checker"]
    assert [event.kind for event in events if isinstance(event, ev.Error)] == ["turn_budget"]
