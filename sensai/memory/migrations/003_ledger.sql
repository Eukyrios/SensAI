-- 003_ledger: the learner ledger, three tables written by agents (#88).
-- Who may write each table is enforced in ledger.py (#90), not here: SQLite has no per-user rights.
BEGIN;

CREATE TABLE errors (
    id         TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    sentence   TEXT NOT NULL,
    correction TEXT NOT NULL,
    error_type TEXT NOT NULL,
    resolved   INTEGER NOT NULL DEFAULT 0 CHECK (resolved IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE srs_cards (
    id            TEXT PRIMARY KEY,
    front         TEXT NOT NULL,
    back          TEXT NOT NULL,
    ease          REAL NOT NULL DEFAULT 2.5,
    interval_days INTEGER NOT NULL DEFAULT 0,
    due_date      TEXT NOT NULL DEFAULT (date('now')),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE facts (
    id         TEXT PRIMARY KEY,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_errors_resolved ON errors(resolved);
CREATE INDEX idx_srs_cards_due_date ON srs_cards(due_date);
CREATE INDEX idx_facts_key ON facts(key);

COMMIT;
