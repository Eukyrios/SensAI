"""Migration runner: applies pending migrations/NNN_*.sql, tracked via PRAGMA user_version (#50)."""
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def migrate(conn: sqlite3.Connection) -> None:
    """Apply every migration file above the DB's user_version, in order. Idempotent."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = int(path.stem.split("_", 1)[0])
        if version <= current:
            continue
        conn.executescript(path.read_text())
        conn.execute(f"PRAGMA user_version = {version}")
        current = version
