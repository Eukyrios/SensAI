"""Persistent memory and context, owned by B.

    db.py           SQLite connection: WAL, foreign keys, ":memory:" for tests   (layer 1)
    migrations/     schema, one file per change, applied by migrate.py          (layer 1)
    migrate.py      migration runner, tracks PRAGMA user_version, idempotent    (layer 1)
    repository.py   SessionRepository (sync) + AsyncSessionRepository facade    (layer 1)
    ledger.py       errors, srs_cards, facts tables with per-agent ownership    (layer 3)
    context.py      context providers and builder with token budget            (layer 4)
    compression.py  sliding-window summarisation                                (layer 5)
    srs.py          spaced-repetition scheduling for the curriculum agent       (layer 6)
"""
