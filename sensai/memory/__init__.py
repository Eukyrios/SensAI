"""Persistent memory and context, owned by B.

    db.py           SQLite connection (WAL, foreign keys) and migrations        (layer 1)
    sessions.py     save/reload sessions, message tree, learner profile        (layer 2)
    ledger.py       errors, srs_cards, facts tables with per-agent ownership   (layer 3)
    context.py      context providers and builder with token budget           (layer 4)
    compression.py  sliding-window summarisation                               (layer 5)
    srs.py          spaced-repetition scheduling for the curriculum agent      (layer 6)
"""
