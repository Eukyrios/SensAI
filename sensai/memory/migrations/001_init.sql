-- 001_init: sessions, messages, turns and agent_runs (#48).
-- Ids are TEXT uuids from sensai.core.ids.new_id(), the same ids the events carry, so a turn
-- can name its turn_id and run_ids before any row is written.
BEGIN;

CREATE TABLE sessions (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    model      TEXT NOT NULL,
    title      TEXT
);

CREATE TABLE messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    parent_id  TEXT REFERENCES messages(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE turns (
    id                    TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_message_id       TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    assistant_message_id  TEXT REFERENCES messages(id) ON DELETE CASCADE,   -- NULL if the reply failed
    latency_ms            INTEGER,
    tokens                INTEGER
);

CREATE TABLE agent_runs (
    id             TEXT PRIMARY KEY,
    turn_id        TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    parent_run_id  TEXT REFERENCES agent_runs(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,
    input          TEXT,
    output         TEXT,
    status         TEXT NOT NULL,
    duration_ms    INTEGER
);

CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_turns_session_id ON turns(session_id);
CREATE INDEX idx_agent_runs_turn_id ON agent_runs(turn_id);

COMMIT;
