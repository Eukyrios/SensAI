-- 002_active_leaf_and_profile: branch pointer on sessions, learner profile table (#67 #68 #69).
BEGIN;

-- The tip of the branch a session is currently on; history is the chain of parent_id from here.
ALTER TABLE sessions ADD COLUMN active_leaf_id TEXT REFERENCES messages(id) ON DELETE SET NULL;

CREATE TABLE learner_profiles (
    name       TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

COMMIT;
