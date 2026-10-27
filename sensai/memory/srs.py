# Spaced-repetition scheduling (SM-2) for the srs_cards ledger table. next_review() is pure, so it is
# tested with fixed dates; the helpers below read and write cards through the ledger.

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from ..knowledge.tools import Tool
from .ledger import Ledger

OWNER = "curriculum"
MIN_EASE = 1.3
GRADES = {"again": 1, "hard": 3, "good": 4, "easy": 5}
FIRST_INTERVAL_DAYS = 1
SECOND_INTERVAL_DAYS = 6


def next_review(ease: float, interval_days: int, grade: int, today: date) -> tuple[float, int, date]:
    """Return the card's new (ease, interval_days, due_date) after an answer graded 0-5."""
    if not 0 <= grade <= 5:
        raise ValueError(f"grade must be between 0 and 5, got {grade}")
    new_ease = max(MIN_EASE, ease + 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    if grade < 3:
        interval = FIRST_INTERVAL_DAYS
    elif interval_days < FIRST_INTERVAL_DAYS:
        interval = FIRST_INTERVAL_DAYS
    elif interval_days < SECOND_INTERVAL_DAYS:
        interval = SECOND_INTERVAL_DAYS
    else:
        interval = round(interval_days * new_ease)
    return round(new_ease, 2), interval, today + timedelta(days=interval)


def due_cards(ledger: Ledger, today: date) -> list:
    return [card for card in ledger.list("srs_cards") if card["due_date"] <= today.isoformat()]


def add_card(ledger: Ledger, front: str, back: str, today: date) -> str | None:
    """Create a card due today, or return None when a card with this front already exists."""
    if ledger.list("srs_cards", front=front):
        return None
    return ledger.create(OWNER, "srs_cards", front=front, back=back, due_date=today.isoformat())


def review(ledger: Ledger, card_id: str, grade: int, today: date) -> sqlite3.Row:
    card = ledger.read("srs_cards", card_id)
    ease, interval, due = next_review(card["ease"], card["interval_days"], grade, today)
    ledger.update(OWNER, "srs_cards", card_id, ease=ease, interval_days=interval, due_date=due.isoformat())
    return ledger.read("srs_cards", card_id)


def _in_exchange(front: str, back: str, exchange: str) -> bool:
    """The word is there as written, or its reading (first word of back, e.g. romaji) is."""
    words = back.replace("—", " ").replace(",", " ").split()
    reading = words[0].lower() if words else ""
    return front in exchange or bool(reading) and reading in exchange.lower()


def tools(ledger: Ledger, today: date, exchange: str | None = None, quizzed: bool = True) -> list[Tool]:
    """The curriculum agent's tools: they apply the scheduling, the model never computes dates.

    Two guards keep a small model honest: a new card's word or reading must appear in the exchange (`exchange`),
    and a card is only graded when the tutor had just asked a question (`quizzed`)."""

    def add_card_tool(front: str, back: str) -> dict:
        if exchange is not None and not _in_exchange(front, back, exchange):
            raise ValueError(f"{front} does not appear in the exchange, so no card was added")
        card_id = add_card(ledger, front, back, today)
        return {"added": card_id} if card_id else {"skipped": f"a card for {front} already exists"}

    def review_tool(front: str, grade: str) -> dict:
        if not quizzed:
            raise ValueError("the tutor asked no question, so the learner answered nothing to grade")
        if grade not in GRADES:
            raise ValueError(f"grade must be one of {', '.join(GRADES)}")
        cards = ledger.list("srs_cards", front=front)
        if not cards:
            raise LookupError(f"no card for {front}")
        card = review(ledger, cards[0]["id"], GRADES[grade], today)
        return {"front": front, "interval_days": card["interval_days"], "due_date": card["due_date"]}

    return [
        Tool("srs_add_card", "Add a vocabulary card for a Japanese word the learner just met. front is the "
             "word in Japanese, back is its reading and English meaning.",
             {"type": "object", "properties": {"front": {"type": "string"}, "back": {"type": "string"}},
              "required": ["front", "back"]}, add_card_tool),
        Tool("srs_review", "Grade a known card the learner answered: again, hard, good or easy. It "
             "reschedules the card.",
             {"type": "object", "properties": {"front": {"type": "string"},
                                               "grade": {"type": "string", "enum": list(GRADES)}},
              "required": ["front", "grade"]}, review_tool),
    ]
