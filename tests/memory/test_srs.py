# Tests for the SRS scheduling, with fixed dates.

from datetime import date

import pytest

from sensai.memory import srs
from sensai.memory.ledger import Ledger, WriteNotAllowed

TODAY = date(2026, 1, 10)


@pytest.fixture
def ledger(conn):
    return Ledger(conn)


def test_first_good_answers_walk_1_then_6_then_ease_multiples():
    ease, interval, due = srs.next_review(2.5, 0, 4, TODAY)
    assert (ease, interval, due) == (2.5, 1, date(2026, 1, 11))
    ease, interval, due = srs.next_review(ease, interval, 4, due)
    assert (interval, due) == (6, date(2026, 1, 17))
    ease, interval, due = srs.next_review(ease, interval, 4, due)
    assert (interval, due) == (15, date(2026, 2, 1))


def test_wrong_answer_brings_the_card_back_and_lowers_ease():
    ease, interval, due = srs.next_review(2.5, 15, 1, TODAY)
    assert interval == 1 and due == date(2026, 1, 11)
    assert ease < 2.5


def test_correct_answer_pushes_the_card_further():
    _, interval, _ = srs.next_review(2.5, 6, 4, TODAY)
    assert interval > 6


def test_ease_never_drops_below_the_floor():
    ease = 1.4
    for _ in range(5):
        ease, _, _ = srs.next_review(ease, 1, 0, TODAY)
    assert ease == srs.MIN_EASE


def test_easy_raises_ease():
    assert srs.next_review(2.5, 6, 5, TODAY)[0] == 2.6


@pytest.mark.parametrize("grade", [-1, 6])
def test_rejects_grades_out_of_range(grade):
    with pytest.raises(ValueError):
        srs.next_review(2.5, 1, grade, TODAY)


def test_add_card_is_due_today_and_not_duplicated(ledger):
    card_id = srs.add_card(ledger, "猫", "neko, cat", TODAY)
    assert ledger.read("srs_cards", card_id)["due_date"] == "2026-01-10"
    assert srs.add_card(ledger, "猫", "neko, cat", TODAY) is None
    assert len(ledger.list("srs_cards")) == 1


def test_due_cards_lists_cards_due_today_or_earlier(ledger):
    srs.add_card(ledger, "猫", "cat", date(2026, 1, 9))
    srs.add_card(ledger, "犬", "dog", date(2026, 1, 10))
    srs.add_card(ledger, "鳥", "bird", date(2026, 1, 11))
    assert [c["front"] for c in srs.due_cards(ledger, TODAY)] == ["猫", "犬"]


def test_review_updates_the_card_in_the_ledger(ledger):
    card_id = srs.add_card(ledger, "猫", "cat", TODAY)
    card = srs.review(ledger, card_id, srs.GRADES["good"], TODAY)
    assert (card["interval_days"], card["due_date"]) == (1, "2026-01-11")
    assert srs.due_cards(ledger, TODAY) == []


def test_only_the_curriculum_agent_writes_cards(ledger):
    with pytest.raises(WriteNotAllowed):
        ledger.create("sensei", "srs_cards", front="猫", back="cat")


def test_tools_add_and_review_cards(ledger):
    add, grade = {tool.name: tool for tool in srs.tools(ledger, TODAY)}.values()
    assert "added" in add.func(front="猫", back="cat")
    assert "skipped" in add.func(front="猫", back="cat")
    assert grade.func(front="猫", grade="good") == {"front": "猫", "interval_days": 1, "due_date": "2026-01-11"}


def test_review_tool_rejects_unknown_cards_and_grades(ledger):
    _, grade = srs.tools(ledger, TODAY)
    with pytest.raises(LookupError):
        grade.func(front="猫", grade="good")
    srs.add_card(ledger, "猫", "cat", TODAY)
    with pytest.raises(ValueError):
        grade.func(front="猫", grade="perfect")


def test_tool_guards_check_the_exchange_and_the_question(ledger):
    add, grade = srs.tools(ledger, TODAY, exchange="猫 is neko", quizzed=False)
    with pytest.raises(ValueError, match="does not appear"):
        add.func(front="犬", back="dog")
    assert "added" in add.func(front="猫", back="cat")
    with pytest.raises(ValueError, match="no question"):
        grade.func(front="猫", grade="easy")


def test_add_card_guard_accepts_the_romaji_reading(ledger):
    add, _ = srs.tools(ledger, TODAY, exchange="Inu [ee-noo] — dog.")
    assert "added" in add.func(front="犬", back="inu — dog")
    with pytest.raises(ValueError):
        add.func(front="猫", back="neko — cat")
