# Tests for run_turn(): event order, history, errors and cancellation.

import asyncio
import threading

import pytest

from sensai.core import events as ev
from sensai.llm.ollama import ChatStats, OllamaError, Thought
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


async def test_cancellation_before_any_token_drops_the_question(session):
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


async def test_turn_finished_carries_ollama_stats():
    stats = ChatStats(prompt_eval_count=35, eval_count=12, total_duration_ns=500_000_000)
    session = Session(make_config(), client=FakeClient(stats=stats))
    events = await collect(session, "hello")
    finished = events[-1]
    assert (finished.prompt_eval_count, finished.eval_count) == (35, 12)
    assert finished.total_duration_ns == 500_000_000
    assert not any(isinstance(e, ev.Token) and not isinstance(e.text, str) for e in events)
    assert session.history[-1]["content"] == "Konnichiwa"


async def test_turn_finished_without_stats_measures_duration(session):
    finished = (await collect(session, "hello"))[-1]
    assert (finished.prompt_eval_count, finished.eval_count) == (0, 0)
    assert finished.total_duration_ns > 0


async def test_stream_tokens_yields_stats_last(session):
    session.client = FakeClient(stats=ChatStats(eval_count=2))
    items = [i async for i in stream_tokens(session.client, session.spec, [])]
    assert items == ["Kon", "nichiwa", ChatStats(eval_count=2)]


async def test_stopped_turn_keeps_question_and_partial_reply(session):
    first_token = threading.Event()

    def slow(model, messages, **options):
        yield "Kon"
        first_token.set()
        threading.Event().wait(2)
        yield "nichiwa"

    session.client.chat_stream = slow
    task = asyncio.ensure_future(collect(session, "hello"))
    await asyncio.to_thread(first_token.wait, 2)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.history == [{"role": "user", "content": "hello"},
                               {"role": "assistant", "content": "Kon"}]


async def test_thinking_becomes_events_and_stays_out_of_the_reply():
    client = FakeClient(tokens=(Thought("Is it "), Thought("a greeting?"), "Kon", "nichiwa"))
    session = Session(make_config(), client=client)
    events = await collect(session, "hello")
    assert kinds(events) == ["TurnStarted", "AgentRunStarted", "Thinking", "Thinking", "Token",
                             "Token", "AgentRunFinished", "TurnFinished"]
    thinking = [e for e in events if isinstance(e, ev.Thinking)]
    assert "".join(t.text for t in thinking) == "Is it a greeting?"
    assert thinking[0].run_id == events[1].run_id
    assert session.history[-1] == {"role": "assistant", "content": "Konnichiwa"}


from tests.conftest import memory_repository


async def test_turn_is_saved_and_reloads_identically(session):
    await collect(session, "hello")
    assert session.repo.load_history(session.id) == session.history == [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "Konnichiwa"}]


async def test_user_message_is_saved_before_the_model_is_called(session):
    seen = []

    def spy(model, messages, **options):
        seen.append(session.repo.load_history(session.id))
        yield "ok"

    session.client.chat_stream = spy
    await collect(session, "hello")
    assert seen == [[{"role": "user", "content": "hello"}]]


async def test_resumed_session_continues_the_same_branch():
    repo = memory_repository()
    first = Session(make_config(), client=FakeClient(), repo=repo)
    await collect(first, "one")
    resumed = Session.resume(make_config(), first.id, client=FakeClient(), repo=repo)
    await collect(resumed, "two")
    assert [m["content"] for m in repo.load_history(first.id)] == [
        "one", "Konnichiwa", "two", "Konnichiwa"]
    assert [m["content"] for m in resumed.client.calls[0][1][1:]] == ["one", "Konnichiwa", "two"]


async def test_profile_is_in_the_prompt_but_never_stored():
    repo = memory_repository()
    repo.save_profile("default", "N5 learner, likes anime")
    session = Session(make_config(), client=FakeClient(), repo=repo)
    await collect(session, "hello")
    messages = session.client.calls[0][1]
    assert messages[0] == {"role": "system", "content": session.spec.system_prompt}
    assert messages[1]["role"] == "system" and "N5 learner" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "hello"}
    stored = [m["content"] for m in repo.list_messages(session.id)]
    assert len(stored) == 2 and not any("N5 learner" in c for c in stored)


async def test_failed_turn_leaves_no_orphan_message_in_the_database(session):
    await collect(session, "first")
    session.client = FakeClient(error=OllamaError("connection", "down"))
    await collect(session, "second")
    assert [m["content"] for m in session.repo.load_history(session.id)] == ["first", "Konnichiwa"]
    session.client = FakeClient()
    await collect(session, "third")
    assert [m["content"] for m in session.repo.load_history(session.id)] == [
        "first", "Konnichiwa", "third", "Konnichiwa"]


async def test_unexpected_error_removes_the_stored_question(session):
    session.client = FakeClient(error=RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        await collect(session, "hello")
    assert session.repo.load_history(session.id) == []


async def test_stopped_turn_stores_question_and_partial_reply(session):
    first_token = threading.Event()

    def slow(model, messages, **options):
        yield "Kon"
        first_token.set()
        threading.Event().wait(2)
        yield "nichiwa"

    session.client.chat_stream = slow
    task = asyncio.ensure_future(collect(session, "hello"))
    await asyncio.to_thread(first_token.wait, 2)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.repo.load_history(session.id) == session.history == [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "Kon"}]


async def test_stopped_turn_without_output_removes_the_question(session):
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
    assert session.repo.load_history(session.id) == []


async def test_abandoned_turn_drops_the_unanswered_question(session):
    turn = run_turn(session, "hello")
    async for event in turn:
        if isinstance(event, ev.Token):
            break                                   # the caller stops reading mid-reply
    await turn.aclose()
    assert session.history == []
