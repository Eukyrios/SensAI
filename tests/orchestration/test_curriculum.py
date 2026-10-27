# Tests for the curriculum pass: the agent calls its tools after the visible answer and never breaks the turn.

import json
from datetime import date

import pytest

from sensai.core import events as ev
from sensai.llm.ollama import OllamaError
from sensai.memory import srs
from sensai.memory.ledger import Ledger
from sensai.orchestration.curriculum import run_curriculum
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import FakeClient, ScriptedClient, make_config

TODAY = date(2026, 1, 10)


def step(tool, **args):
    return json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


DONE = step("done")
CAT_ANSWER = "猫 (neko) means cat."
QUESTION = "猫 (neko) — cat. Can you say it back?"
ADD_CAT = step("srs_add_card", front="猫", back="neko, cat")


def session_with(*replies, **config):
    config = make_config(curriculum={"enabled": True, "max_steps": 8, **config})
    return Session(config, client=ScriptedClient(*replies))


def cards(session):
    return [dict(card) for card in Ledger(session.repo.conn).list("srs_cards")]


async def collect(session, text):
    return [e async for e in run_turn(session, text)]


async def test_agent_adds_cards_through_its_tool_after_the_answer():
    session = session_with(CAT_ANSWER, ADD_CAT, DONE)
    events = await collect(session, "What is cat?")
    assert [c["front"] for c in cards(session)] == ["猫"]
    assert type(events[-1]).__name__ == "TurnFinished"
    assert len(session.client.calls) == 3


async def test_curriculum_run_emits_no_event_to_the_surface():
    events = await collect(session_with(CAT_ANSWER, ADD_CAT, DONE), "What is cat?")
    assert {e.agent for e in events if isinstance(e, ev.AgentRunStarted)} == {"sensei"}


async def test_agent_sees_the_exchange_its_tools_and_each_result():
    session = session_with(CAT_ANSWER, ADD_CAT, DONE)
    await collect(session, "What is cat?")
    first, second = session.client.calls[1][1], session.client.calls[2][1]
    assert "srs_add_card" in first[0]["content"] and "srs_review" in first[0]["content"]
    assert CAT_ANSWER in first[-2]["content"]
    assert second[-1]["content"].startswith("Tool result: ")
    assert '"added"' in second[-1]["content"]


async def test_agent_reviews_a_known_card_after_a_question():
    session = session_with("Right! 猫 is neko.", step("srs_review", front="猫", grade="good"), DONE)
    srs.add_card(Ledger(session.repo.conn), "猫", "cat", date(2026, 1, 1))
    session.history += [{"role": "assistant", "content": QUESTION}]
    await collect(session, "neko")
    assert cards(session)[0]["interval_days"] == 1


async def test_agent_cannot_grade_a_card_when_no_question_was_asked():
    session = session_with("猫 (neko) — cat.", step("srs_review", front="猫", grade="easy"), DONE)
    srs.add_card(Ledger(session.repo.conn), "猫", "cat", date(2026, 1, 1))
    await collect(session, "Quiz me on 猫")
    assert cards(session)[0]["interval_days"] == 0
    assert "no question" in session.client.calls[2][1][-1]["content"]


async def test_agent_cannot_add_a_word_that_is_not_in_the_exchange():
    session = session_with("Hello!", step("srs_add_card", front="犬", back="inu, dog"), DONE)
    await collect(session, "Hi")
    assert cards(session) == []
    assert "does not appear" in session.client.calls[2][1][-1]["content"]


async def test_a_failing_tool_reports_its_error_and_the_agent_goes_on():
    session = session_with("Right!", step("srs_review", front="犬", grade="good"), DONE)
    session.history += [{"role": "assistant", "content": "犬 (inu)?"}]
    await collect(session, "inu")
    assert "no card for 犬" in session.client.calls[2][1][-1]["content"]


async def test_loop_stops_after_max_steps():
    session = session_with("猫 hi.", ADD_CAT, ADD_CAT, ADD_CAT, max_steps=2)
    await collect(session, "Hi")
    assert len(session.client.calls) == 3
    assert len(cards(session)) == 1


async def test_disabled_curriculum_never_calls_the_model():
    session = Session(make_config(), client=FakeClient())
    await collect(session, "Hi")
    assert len(session.client.calls) == 1


async def test_failed_turn_skips_the_curriculum():
    session = Session(make_config(curriculum={"enabled": True}),
                      client=FakeClient(error=OllamaError("stream", "boom")))
    await collect(session, "Hi")
    assert cards(session) == []


async def test_invalid_curriculum_output_is_swallowed():
    session = session_with("Hello.", "not json", "still not json")
    events = await collect(session, "Hi")
    assert type(events[-1]).__name__ == "TurnFinished"
    assert not any(isinstance(e, ev.Error) for e in events)
    assert cards(session) == []


async def test_curriculum_crash_does_not_break_the_turn():
    session = session_with("Hello.")
    events = await collect(session, "Hi")
    assert type(events[-1]).__name__ == "TurnFinished"


async def test_run_curriculum_without_history_does_nothing():
    session = session_with()
    await run_curriculum(session, "t1", TODAY)
    assert session.client.calls == []
