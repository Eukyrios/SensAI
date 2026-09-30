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
from . import db
from .migrate import migrate


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
        message_id = message_id or new_id()
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (id, session_id, role, content, parent_id) VALUES (?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, parent_id),
            )
        return message_id

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

    async def record_turn(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.repo.record_turn, *args, **kwargs)

    async def record_agent_run(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.repo.record_agent_run, *args, **kwargs)


def open_repository(config: dict) -> SessionRepository:
    """Connect using config['storage']['db_path'], apply pending migrations, return a repository."""
    conn = db.connect(config["storage"]["db_path"])
    migrate(conn)
    return SessionRepository(conn)


def open_async_repository(config: dict) -> AsyncSessionRepository:
    return AsyncSessionRepository(open_repository(config))
