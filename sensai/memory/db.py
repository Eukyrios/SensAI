# SQLite connection: WAL journal, enforced foreign keys, row objects, serialised across threads.

import sqlite3
import threading

from ..core.config import resolve


class LockedCursor(sqlite3.Cursor):

    def execute(self, *args):
        with self.connection.lock:
            return super().execute(*args)

    def executemany(self, *args):
        with self.connection.lock:
            return super().executemany(*args)

    def fetchone(self):
        with self.connection.lock:
            return super().fetchone()

    def fetchmany(self, *args):
        with self.connection.lock:
            return super().fetchmany(*args)

    def fetchall(self):
        with self.connection.lock:
            return super().fetchall()


class LockedConnection(sqlite3.Connection):
    """One shared connection, guarded by a reentrant lock.

    `with conn:` holds the lock for the whole transaction, so another thread
    can never interleave statements or commit a half-finished transaction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()

    def __enter__(self):
        self.lock.acquire()
        try:
            return super().__enter__()
        except BaseException:
            self.lock.release()
            raise

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.lock.release()

    def cursor(self, factory=LockedCursor):
        return super().cursor(factory)

    def execute(self, *args):
        return self.cursor().execute(*args)

    def executemany(self, *args):
        return self.cursor().executemany(*args)

    def executescript(self, script):
        with self.lock:
            return super().executescript(script)


def connect(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        db_path = resolve(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        path = str(db_path)

    conn = sqlite3.connect(path, check_same_thread=False, factory=LockedConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn
