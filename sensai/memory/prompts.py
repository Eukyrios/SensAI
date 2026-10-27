# Prompt store: every system prompt an agent ran with, kept by content hash, and the agent runs
# grouped by that hash, so two versions of a prompt can be compared and an agent pinned back to one.

import sqlite3
from dataclasses import replace

from ..core.agent_spec import AgentSpec, SpecError, prompt_hash


class PromptStore:

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def resolve(self, spec: AgentSpec) -> AgentSpec:
        if spec.prompt_version:
            return replace(spec, system_prompt=self.get(spec.name, spec.prompt_version))
        self.save(spec.name, spec.system_prompt)
        return spec

    def save(self, agent: str, content: str) -> str:
        content_hash = prompt_hash(content)
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO prompts (hash, agent, content) VALUES (?, ?, ?)",
                (content_hash, agent, content),
            )
        return content_hash

    def get(self, agent: str, content_hash: str) -> str:
        row = self.conn.execute("SELECT content FROM prompts WHERE hash = ?", (content_hash,)).fetchone()
        if row is None:
            raise SpecError(f"agent '{agent}': unknown prompt version {content_hash}")
        return row["content"]

    def versions(self, agent: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT hash, created_at FROM prompts WHERE agent = ? ORDER BY rowid", (agent,)
        ).fetchall()

    def metrics(self, agent: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT prompt_hash, COUNT(*) AS runs, SUM(status = 'ok') AS ok_runs, "
            "AVG(duration_ms) AS avg_duration_ms, AVG(tokens) AS avg_tokens "
            "FROM agent_runs WHERE agent = ? AND prompt_hash IS NOT NULL "
            "GROUP BY prompt_hash ORDER BY MIN(rowid)",
            (agent,),
        ).fetchall()
