-- 001_init: sessions, messages, turns and agent_runs (#48).
BEGIN;

CREATE TABLE sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    model      TEXT NOT NULL,
    title      TEXT
);

CREATE TABLE messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    parent_id  INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE turns (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_message_id       INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    assistant_message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    latency_ms            INTEGER,
    tokens                INTEGER
);

CREATE TABLE agent_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id        INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    parent_run_id  INTEGER REFERENCES agent_runs(id) ON DELETE CASCADE,
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
