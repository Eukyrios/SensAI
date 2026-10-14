# SQLite connection: WAL journal, enforced foreign keys, row objects.

import sqlite3

from ..core.config import resolve


def connect(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        db_path = resolve(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        path = str(db_path)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn
