# Tests for switching persona mid-session: the history stays, the next prompt changes, each is versioned.

import pytest

from sensai.core.agent_spec import SpecError
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import FakeClient, make_config


async def collect(session, text):
    return [event async for event in run_turn(session, text)]


def system_prompt_of_call(session, number):
    return session.client.calls[number][1][0]["content"]


async def test_the_session_starts_with_the_default_persona():
    session = Session(make_config(), client=FakeClient())
    await collect(session, "hello")
    assert session.persona.name == "teacher" and "Persona: teacher." in system_prompt_of_call(session, 0)


async def test_switching_persona_keeps_the_history_and_changes_the_prompt():
    session = Session(make_config(), client=FakeClient())
    await collect(session, "hello")
    session.set_persona("casual")
    await collect(session, "what did I just say?")
    messages = session.client.calls[1][1]
    assert "Persona: casual." in messages[0]["content"] and "Persona: teacher." not in messages[0]["content"]
    assert [message["content"] for message in messages[1:]] == ["hello", "Konnichiwa", "what did I just say?"]
    assert len(session.history) == 4


async def test_each_persona_is_a_prompt_version():
    session = Session(make_config(), client=FakeClient())
    await collect(session, "hello")
    session.set_persona("strict")
    await collect(session, "hello again")
    hashes = [row["prompt_hash"] for row in session.repo.conn.execute("SELECT prompt_hash FROM agent_runs")]
    assert len(set(hashes)) == 2 and len(session.prompts.versions("sensei")) == 2


def test_an_unknown_persona_keeps_the_current_one():
    session = Session(make_config(), client=FakeClient())
    with pytest.raises(SpecError):
        session.set_persona("nobody")
    assert session.persona.name == "teacher"
