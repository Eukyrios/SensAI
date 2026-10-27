-- 004_prompt_versions: every system prompt an agent ran with, kept by content hash, and on each
-- agent run the agent, the hash of its prompt and its tokens, so runs can be compared by prompt (A4).
BEGIN;

CREATE TABLE prompts (
    hash       TEXT PRIMARY KEY,
    agent      TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

ALTER TABLE agent_runs ADD COLUMN agent TEXT;
ALTER TABLE agent_runs ADD COLUMN prompt_hash TEXT REFERENCES prompts(hash);
ALTER TABLE agent_runs ADD COLUMN tokens INTEGER;

CREATE INDEX idx_prompts_agent ON prompts(agent);
CREATE INDEX idx_agent_runs_prompt_hash ON agent_runs(prompt_hash);

COMMIT;
