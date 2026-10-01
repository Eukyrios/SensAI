import asyncio
import threading

import pytest

from sensai.core import events as ev
from sensai.llm.ollama import OllamaError
from sensai.orchestration import turn as turn_module
from sensai.orchestration.session import Session
from sensai.orchestration.turn import build_context, run_turn, stream_tokens
from tests.fakes import FakeClient, make_config


@pytest.fixture
def session():
    return Session(make_config(), client=FakeClient())


async def collect(session, text):
    return [e async for e in run_turn(session, text)]


def kinds(events):
    return [type(e).__name__ for e in events]


async def test_event_order_and_ids(session):
    events = await collect(session, "  hello  ")
    assert kinds(events) == ["TurnStarted", "AgentRunStarted", "Token", "Token",
                             "AgentRunFinished", "TurnFinished"]
    turn_ids = {e.turn_id for e in events}
    assert len(turn_ids) == 1 and len(turn_ids.pop()) == 32
    started, *_, finished = events
    assert started.session_id == session.id and started.user_input == "  hello  "
    runs = {e.run_id for e in events if hasattr(e, "run_id")}
    assert len(runs) == 1
    assert events[1].agent == "sensei" and events[1].parent_run_id is None
    assert "".join(e.text for e in events if isinstance(e, ev.Token)) == "Konnichiwa"
    assert events[-2].ok is True
    assert finished.model == session.spec.model
    assert finished.ttft_ms is not None and finished.total_duration_ns > 0


async def test_history_and_model_call(session):
    await collect(session, "hello")
    assert session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Konnichiwa"},
    ]
    model, messages, options = session.client.calls[0]
    spec = session.spec
    assert model == spec.model
    assert options == {"num_ctx": spec.num_ctx, "temperature": spec.temperature, "think": spec.think}
    assert messages == [{"role": "system", "content": spec.system_prompt},
                        {"role": "user", "content": "hello"}]


async def test_empty_input(session):
    events = await collect(session, "   ")
    assert kinds(events) == ["TurnStarted", "Error", "TurnFinished"]
    assert events[1].kind == "empty_input"
    assert session.client.calls == [] and session.history == []


async def test_model_error_becomes_events_and_rolls_back(session):
    session.client = FakeClient(error=OllamaError("connection", "down"))
    events = await collect(session, "hello")
    assert kinds(events) == ["TurnStarted", "AgentRunStarted", "Error", "AgentRunFinished",
                             "TurnFinished"]
    assert (events[2].kind, events[2].message) == ("connection", "down")
    assert events[3].ok is False and events[-1].ttft_ms is None
    assert session.history == []


async def test_unexpected_error_propagates_and_rolls_back(session):
    session.client = FakeClient(error=RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        await collect(session, "hello")
    assert session.history == []


async def test_guardrail_change_is_reported(session, monkeypatch):
    monkeypatch.setattr(turn_module.guardrails, "filter_input", lambda text, config: "[masked]")
    events = await collect(session, "my email is a@b.c")
    g = next(e for e in events if isinstance(e, ev.GuardrailTriggered))
    assert (g.side, g.action) == ("input", "redact")
    assert session.history[0]["content"] == "[masked]"


async def test_history_is_truncated_in_context():
    session = Session(make_config(memory={"max_history_turns": 1}), client=FakeClient())
    for i in range(3):
        await collect(session, f"m{i}")
    assert [m["content"] for m in session.client.calls[-1][1][1:]] == ["Konnichiwa", "m2"]
    assert build_context(session)[0]["role"] == "system"


async def test_stream_stops_the_producer_when_consumer_leaves(session):
    closed = threading.Event()
    release = threading.Event()

    def endless(model, messages, **options):
        try:
            while True:
                release.wait(1)
                yield "x"
        finally:
            closed.set()

    session.client.chat_stream = endless
    stream = stream_tokens(session.client, session.spec, [])
    release.set()
    assert await stream.__anext__() == "x"
    await stream.aclose()
    assert await asyncio.to_thread(closed.wait, 2)


async def test_cancellation_rolls_back_history(session):
    gate = threading.Event()

    def slow(model, messages, **options):
        gate.wait(2)
        yield "late"

    session.client.chat_stream = slow
    task = asyncio.ensure_future(collect(session, "hello"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    assert session.history == []
