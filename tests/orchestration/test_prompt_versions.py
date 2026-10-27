# Tests for prompt versioning in a turn: each agent run records its prompt hash, and a pin rolls back.

from dataclasses import replace

import pytest

from sensai.core import events as ev
from sensai.core.agent_spec import SpecError
from sensai.llm.ollama import ChatStats, OllamaError
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import FakeClient, ScriptedClient, make_config
from tests.orchestration.test_orchestrator import EXPLANATION, WRONG, checker_session

STATS = ChatStats(prompt_eval_count=30, eval_count=10, total_duration_ns=2_000_000)


async def collect(session, text="How do I say hello?"):
    return [event async for event in run_turn(session, text)]


def agent_runs(session):
    return session.repo.conn.execute("SELECT * FROM agent_runs ORDER BY rowid").fetchall()


def sent_spec(session):
    return session.persona.apply(session.spec)


def with_prompt(session, system_prompt):
    session.spec = replace(session.spec, system_prompt=system_prompt)


async def test_a_kept_turn_records_its_run_with_the_prompt_hash():
    session = Session(make_config(), client=FakeClient(stats=STATS))
    events = await collect(session)
    (run,) = agent_runs(session)
    assert (run["agent"], run["kind"], run["status"]) == ("sensei", "agent", "ok")
    assert run["prompt_hash"] == sent_spec(session).prompt_hash
    assert (run["duration_ms"], run["tokens"], run["output"]) == (2, 40, "Konnichiwa")
    assert run["turn_id"] == events[0].turn_id


async def test_a_kept_turn_records_the_turn():
    session = Session(make_config(), client=FakeClient(stats=STATS))
    await collect(session)
    turn = session.repo.conn.execute("SELECT * FROM turns").fetchone()
    user_message, answer = session.repo.list_messages(session.id)
    assert (turn["user_message_id"], turn["assistant_message_id"]) == (user_message["id"], answer["id"])
    assert turn["tokens"] == 40 and turn["latency_ms"] >= 0


async def test_the_prompt_is_stored_by_its_hash():
    session = Session(make_config(), client=FakeClient())
    await collect(session)
    assert session.prompts.get("sensei", sent_spec(session).prompt_hash) == sent_spec(session).system_prompt


async def test_run_started_event_carries_the_prompt_hash():
    session = Session(make_config(), client=FakeClient())
    events = await collect(session)
    started = next(event for event in events if isinstance(event, ev.AgentRunStarted))
    assert started.prompt_hash == sent_spec(session).prompt_hash


async def test_editing_the_prompt_records_a_new_hash():
    session = Session(make_config(), client=FakeClient())
    await collect(session)
    with_prompt(session, "Answer in one sentence.")
    await collect(session)
    first, second = agent_runs(session)
    assert first["prompt_hash"] != second["prompt_hash"]
    assert len(session.prompts.versions("sensei")) == 2


async def test_metrics_compare_two_prompt_versions():
    session = Session(make_config(), client=FakeClient(stats=STATS))
    await collect(session)
    with_prompt(session, "Answer in one sentence.")
    await collect(session)
    rows = session.prompts.metrics("sensei")
    assert [row["runs"] for row in rows] == [1, 1]
    assert [row["prompt_hash"] for row in rows] == [run["prompt_hash"] for run in agent_runs(session)]


async def test_a_pinned_agent_runs_its_previous_prompt():
    session = Session(make_config(), client=FakeClient())
    original_prompt, original_hash = sent_spec(session).system_prompt, sent_spec(session).prompt_hash
    await collect(session)
    with_prompt(session, "Answer in one sentence.")
    session.spec = replace(session.spec, prompt_version=original_hash)
    await collect(session)
    model, messages, options = session.client.calls[-1]
    assert messages[0] == {"role": "system", "content": original_prompt}
    assert agent_runs(session)[-1]["prompt_hash"] == original_hash


async def test_an_unknown_pin_fails_the_turn_and_drops_the_question():
    session = Session(make_config(), client=FakeClient())
    session.spec = replace(session.spec, prompt_version="000000000000")
    with pytest.raises(SpecError):
        await collect(session)
    assert session.history == [] and agent_runs(session) == []


async def test_a_failed_turn_records_no_run():
    session = Session(make_config(), client=FakeClient(error=OllamaError("connection", "down")))
    await collect(session)
    assert agent_runs(session) == []
    assert session.prompts.versions("sensei") != []


async def test_a_handoff_records_both_runs_under_each_other():
    session = checker_session(ScriptedClient(WRONG, EXPLANATION))
    await collect(session, "私は学校を行きます")
    checker_run, sensei_run = agent_runs(session)
    assert (checker_run["agent"], sensei_run["agent"]) == ("checker", "sensei")
    assert sensei_run["parent_run_id"] == checker_run["id"]
    assert checker_run["prompt_hash"] != sensei_run["prompt_hash"]
