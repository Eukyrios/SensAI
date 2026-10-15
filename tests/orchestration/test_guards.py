# Tests for the turn guards: max hops, handoff cycles, the turn budget and each agent's timeout.

import json
import threading

from sensai.core import events as ev
from sensai.core.agent_spec import AgentSpec
from sensai.core.handoff import Handoff
from sensai.orchestration.guards import TurnGuards
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import FakeClient, ScriptedClient, make_config

SETTINGS = {"max_hops": 3, "turn_budget_s": 100}
LOOPING_SPEC = """name = "{name}"
model = "qwen3.5:9b"
num_ctx = 2048
timeout_s = {timeout_s}
handoffs = ["{target}"]

[output_schema]
type = "object"
required = ["note"]

[output_schema.properties.note]
type = "string"
"""


class Clock:

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class StallingClient(FakeClient):

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, messages, options))
        self.release.wait(5)
        yield "late"


def handoff(to_agent, hop=1):
    return Handoff("ping", to_agent, "loop", {}, hop=hop, run_id="r1")


def handoff_reply(target):
    return json.dumps({"note": "again", "handoff": {"to": target, "reason": "loop"}})


def write_looping_agents(agents_dir, timeout_s=60):
    for name, target in (("ping", "pong"), ("pong", "ping")):
        spec = LOOPING_SPEC.format(name=name, target=target, timeout_s=timeout_s)
        (agents_dir / f"{name}.toml").write_text(spec, encoding="utf-8")


def looping_session(agents_dir, client, **orchestration):
    config = make_config(agents={"dir": str(agents_dir), "default": "ping"},
                         orchestration={**SETTINGS, **orchestration})
    return Session(config, client=client)


async def collect(session, text="hello"):
    return [event async for event in run_turn(session, text)]


def kinds(events):
    return [type(event).__name__ for event in events]


def test_guards_let_a_new_agent_through():
    assert TurnGuards(SETTINGS).stop_before(handoff("pong"), ["ping"]) is None


def test_guards_stop_too_many_hops():
    stop = TurnGuards(SETTINGS).stop_before(handoff("other", hop=4), ["ping"])
    assert stop.kind == "max_hops" and "3 handoffs" in stop.message


def test_guards_stop_an_agent_receiving_the_turn_twice():
    stop = TurnGuards(SETTINGS).stop_before(handoff("ping", hop=2), ["ping", "pong"])
    assert stop.kind == "handoff_cycle" and "ping → pong" in stop.message


def test_guards_stop_a_handoff_after_the_budget():
    clock = Clock()
    guards = TurnGuards(SETTINGS, clock)
    clock.now += 100
    assert guards.stop_before(handoff("pong"), ["ping"]).kind == "turn_budget"


def test_run_deadline_is_the_earliest_of_timeout_and_budget():
    clock = Clock()
    guards = TurnGuards(SETTINGS, clock)
    deadline, stop = guards.run_deadline(AgentSpec(name="quick", model="m", num_ctx=1, timeout_s=10))
    assert deadline == 1010.0 and stop.kind == "timeout" and "quick took longer than 10s" in stop.message
    deadline, stop = guards.run_deadline(AgentSpec(name="slow", model="m", num_ctx=1, timeout_s=500))
    assert deadline == 1100.0 and stop.kind == "turn_budget"
    clock.now += 200
    assert guards.seconds_left(deadline) == 0.0


async def test_a_handoff_loop_is_stopped(tmp_path):
    write_looping_agents(tmp_path)
    session = looping_session(tmp_path, ScriptedClient(handoff_reply("pong"), handoff_reply("ping")))
    events = await collect(session)
    assert [event.agent for event in events if isinstance(event, ev.AgentRunStarted)] == ["ping", "pong"]
    assert kinds(events)[-3:] == ["HandoffEvent", "Error", "TurnFinished"]
    assert events[-2].kind == "handoff_cycle"
    assert len(session.client.calls) == 2 and session.history == []


async def test_max_hops_stops_before_the_cycle(tmp_path):
    write_looping_agents(tmp_path)
    session = looping_session(tmp_path, ScriptedClient(handoff_reply("pong"), handoff_reply("ping")),
                              max_hops=1)
    events = await collect(session)
    assert events[-2].kind == "max_hops"


async def test_a_slow_agent_is_timed_out(tmp_path):
    write_looping_agents(tmp_path, timeout_s=0.2)
    session = looping_session(tmp_path, StallingClient())
    events = await collect(session)
    session.client.release.set()
    assert kinds(events) == ["TurnStarted", "AgentRunStarted", "Error", "AgentRunFinished", "TurnFinished"]
    assert events[2].kind == "timeout" and events[3].ok is False
    assert session.history == []


async def test_the_turn_budget_stops_the_run():
    session = Session(make_config(orchestration={**SETTINGS, "turn_budget_s": 0.2}),
                      client=StallingClient())
    events = await collect(session)
    session.client.release.set()
    assert events[2].kind == "turn_budget" and "0.2s budget" in events[2].message
    assert session.history == []
