# Session storage: a synchronous repository plus an async facade, sessions, messages and profiles.

import asyncio
import sqlite3

from ..core.ids import new_id
from ..core.messages import Message
from . import db
from .ledger import Ledger
from .migrate import migrate


class SessionNotFound(LookupError):
    pass


class SessionRepository:

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_session(self, model: str, title: str | None = None, *,
                       session_id: str | None = None) -> str:
        session_id = session_id or new_id()
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions (id, model, title) VALUES (?, ?, ?)", (session_id, model, title)
            )
        return session_id

    def add_message(self, session_id: str, role: str, content: str,
                    parent_id: str | None = None, *, message_id: str | None = None) -> str:
        message_id = message_id or new_id()
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (id, session_id, role, content, parent_id) VALUES (?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, parent_id),
            )
            self.conn.execute(
                "UPDATE sessions SET active_leaf_id = ? WHERE id = ?", (message_id, session_id)
            )
        return message_id

    def discard_leaf(self, message_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET active_leaf_id = "
                "(SELECT parent_id FROM messages WHERE id = ?) WHERE active_leaf_id = ?",
                (message_id, message_id),
            )
            self.conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    def get_session(self, session_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFound(session_id)
        return row

    def list_sessions(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM sessions ORDER BY rowid DESC").fetchall()

    def set_active_leaf(self, session_id: str, message_id: str) -> None:
        self.get_session(session_id)
        with self.conn:
            cur = self.conn.execute(
                "UPDATE sessions SET active_leaf_id = ? WHERE id = ? AND EXISTS "
                "(SELECT 1 FROM messages WHERE id = ? AND session_id = ?)",
                (message_id, session_id, message_id, session_id),
            )
        if cur.rowcount == 0:
            raise SessionNotFound(f"message {message_id} is not in session {session_id}")

    def load_history(self, session_id: str) -> list[Message]:
        leaf = self.get_session(session_id)["active_leaf_id"]
        rows = self.conn.execute(
            "WITH RECURSIVE branch(id, parent_id, role, content, depth) AS ("
            " SELECT id, parent_id, role, content, 0 FROM messages WHERE id = ?"
            " UNION ALL"
            " SELECT m.id, m.parent_id, m.role, m.content, b.depth + 1"
            " FROM messages m JOIN branch b ON m.id = b.parent_id)"
            " SELECT role, content FROM branch ORDER BY depth DESC",
            (leaf,),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def get_profile(self, name: str = "default") -> str | None:
        row = self.conn.execute(
            "SELECT content FROM learner_profiles WHERE name = ?", (name,)
        ).fetchone()
        return row["content"] if row else None

    def save_profile(self, name: str, content: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO learner_profiles (name, content) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET content = excluded.content, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                (name, content),
            )

    def profile_system_message(self, name: str = "default") -> Message | None:
        content = self.get_profile(name)
        if not content:
            return None
        return {"role": "system", "content": f"Learner profile:\n{content}"}

    def list_messages(self, session_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY rowid", (session_id,)
        ).fetchall()

    def record_turn(self, session_id: str, user_message_id: str,
                    assistant_message_id: str | None = None,
                    latency_ms: int | None = None, tokens: int | None = None, *,
                    turn_id: str | None = None) -> str:
        turn_id = turn_id or new_id()
        with self.conn:
            self.conn.execute(
                "INSERT INTO turns (id, session_id, user_message_id, assistant_message_id, "
                "latency_ms, tokens) VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, session_id, user_message_id, assistant_message_id, latency_ms, tokens),
            )
        return turn_id

    def record_agent_run(self, turn_id: str, kind: str, status: str,
                         input: str | None = None, output: str | None = None,
                         duration_ms: int | None = None,
                         parent_run_id: str | None = None, *, run_id: str | None = None,
                         agent: str | None = None, prompt_hash: str | None = None,
                         tokens: int | None = None) -> str:
        run_id = run_id or new_id()
        with self.conn:
            self.conn.execute(
                "INSERT INTO agent_runs (id, turn_id, kind, input, output, status, duration_ms, "
                "parent_run_id, agent, prompt_hash, tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, turn_id, kind, input, output, status, duration_ms, parent_run_id,
                 agent, prompt_hash, tokens),
            )
        return run_id


class AsyncSessionRepository:

    def __init__(self, repo: SessionRepository):
        self.repo = repo

    async def create_session(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.repo.create_session, *args, **kwargs)

    async def add_message(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.repo.add_message, *args, **kwargs)

    async def list_messages(self, *args, **kwargs) -> list[sqlite3.Row]:
        return await asyncio.to_thread(self.repo.list_messages, *args, **kwargs)

    async def discard_leaf(self, *args, **kwargs) -> None:
        return await asyncio.to_thread(self.repo.discard_leaf, *args, **kwargs)

    async def get_session(self, *args, **kwargs) -> sqlite3.Row:
        return await asyncio.to_thread(self.repo.get_session, *args, **kwargs)

    async def list_sessions(self, *args, **kwargs) -> list[sqlite3.Row]:
        return await asyncio.to_thread(self.repo.list_sessions, *args, **kwargs)

    async def set_active_leaf(self, *args, **kwargs) -> None:
        return await asyncio.to_thread(self.repo.set_active_leaf, *args, **kwargs)

    async def load_history(self, *args, **kwargs) -> list[Message]:
        return await asyncio.to_thread(self.repo.load_history, *args, **kwargs)

    async def get_profile(self, *args, **kwargs) -> str | None:
        return await asyncio.to_thread(self.repo.get_profile, *args, **kwargs)

    async def save_profile(self, *args, **kwargs) -> None:
        return await asyncio.to_thread(self.repo.save_profile, *args, **kwargs)

    async def profile_system_message(self, *args, **kwargs) -> Message | None:
        return await asyncio.to_thread(self.repo.profile_system_message, *args, **kwargs)

    async def record_turn(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.repo.record_turn, *args, **kwargs)

    async def record_agent_run(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.repo.record_agent_run, *args, **kwargs)


def open_repository(config: dict) -> SessionRepository:
    conn = db.connect(config["storage"]["db_path"])
    migrate(conn)
    return SessionRepository(conn)


def open_ledger(config: dict) -> Ledger:
    conn = db.connect(config["storage"]["db_path"])
    migrate(conn)
    return Ledger(conn)


def open_async_repository(config: dict) -> AsyncSessionRepository:
    return AsyncSessionRepository(open_repository(config))
