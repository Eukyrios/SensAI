"""Session storage: a synchronous repository plus an async facade for FastAPI (#51).

sqlite3 is blocking, which would stall an async web server. Two ways to bridge that:
aiosqlite (a small wrapper that runs the same blocking calls on a background thread) or
asyncio.to_thread around a plain synchronous repository. We picked asyncio.to_thread: it
adds no dependency (this project avoids frameworks it doesn't need — see requirements.txt),
keeps all the storage logic in one place that stays trivial to unit-test synchronously, and
the thread-pool overhead is irrelevant at Sensai's scale (one learner, a handful of DB calls
per turn). See the README's Persistence section.
"""
import asyncio
import sqlite3

from ..core.ids import new_id
from ..core.messages import Message
from . import db
from .ledger import Ledger
from .migrate import migrate


class SessionNotFound(LookupError):
    """No session with this id (or the message is not part of it)."""


class SessionRepository:
    """Synchronous CRUD over sessions, messages, turns and agent_runs.

    Ids are text uuids. Pass the id the events already use (e.g. the turn_id from TurnStarted),
    or leave it out and one is generated. Every write returns the row's id.
    """

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
        """Insert a message and move the session's active leaf onto it, atomically."""
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
        """Delete a message nobody replied to, and step the active leaf back onto its parent."""
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
        """All sessions, most recent first (rowid: uuids are random, so they can't order rows)."""
        return self.conn.execute("SELECT * FROM sessions ORDER BY rowid DESC").fetchall()

    def set_active_leaf(self, session_id: str, message_id: str) -> None:
        """Switch the session to the branch ending at `message_id`."""
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
        """Messages from the root down to the session's active leaf, oldest first."""
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
        return [{"role": r["role"], "content": r["content"]} for r in rows]

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
        """The profile as a system message to prepend to the prompt; never stored in messages."""
        content = self.get_profile(name)
        if not content:
            return None
        return {"role": "system", "content": f"Learner profile:\n{content}"}

    def list_messages(self, session_id: str) -> list[sqlite3.Row]:
        """Messages in insertion order (rowid: uuids are random, so they can't order rows)."""
        return self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY rowid", (session_id,)
        ).fetchall()

    def record_turn(self, session_id: str, user_message_id: str,
                    assistant_message_id: str | None = None,
                    latency_ms: int | None = None, tokens: int | None = None, *,
                    turn_id: str | None = None) -> str:
        """Record a turn. assistant_message_id is None when the reply failed."""
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
                         parent_run_id: str | None = None, *, run_id: str | None = None) -> str:
        run_id = run_id or new_id()
        with self.conn:
            self.conn.execute(
                "INSERT INTO agent_runs (id, turn_id, kind, input, output, status, duration_ms, "
                "parent_run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, turn_id, kind, input, output, status, duration_ms, parent_run_id),
            )
        return run_id


class AsyncSessionRepository:
    """Async facade: runs each SessionRepository call in a worker thread."""

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
    """Connect using config['storage']['db_path'], apply pending migrations, return a repository."""
    conn = db.connect(config["storage"]["db_path"])
    migrate(conn)
    return SessionRepository(conn)


def open_ledger(config: dict) -> Ledger:
    """Same database as the repository, for the errors, srs_cards and facts tables."""
    conn = db.connect(config["storage"]["db_path"])
    migrate(conn)
    return Ledger(conn)


def open_async_repository(config: dict) -> AsyncSessionRepository:
    return AsyncSessionRepository(open_repository(config))
