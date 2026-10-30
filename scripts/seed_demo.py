# Demo database: builds a fresh SQLite database from scripts/demo_seed.toml (learner profile, past
# sessions and the learner ledger), so the keynote never depends on typing a whole session live.
# Run `python -m scripts.seed_demo`, then start Sensai with SENSAI_STORAGE__DB_PATH=data/demo.db.

import argparse
from datetime import date, timedelta
from pathlib import Path

from sensai.core.config import load_toml, resolve
from sensai.memory import db
from sensai.memory.ledger import DEFAULT_OWNERS, Ledger
from sensai.memory.migrate import migrate
from sensai.memory.repository import SessionRepository

SEED_PATH = resolve("scripts/demo_seed.toml")
DB_PATH = resolve("data/demo.db")
MODEL = "qwen3.5:9b"


def seed(db_path: Path, seed_path: Path = SEED_PATH, today: date | None = None) -> SessionRepository:
    remove_database(db_path)
    conn = db.connect(str(db_path))
    migrate(conn)
    repo = SessionRepository(conn)
    data = load_toml(seed_path)
    repo.save_profile("default", data["profile"])
    for session in data["sessions"]:
        add_session(repo, session)
    fill_ledger(Ledger(conn), data, today or date.today())
    return repo


def remove_database(db_path: Path) -> None:
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        path.unlink(missing_ok=True)


def add_session(repo: SessionRepository, session: dict) -> str:
    session_id = repo.create_session(MODEL, session["title"], session_id=session.get("id"))
    parent_id = None
    for message in session["messages"]:
        parent_id = repo.add_message(session_id, message["role"], message["content"], parent_id)
    return session_id


def fill_ledger(ledger: Ledger, data: dict, today: date) -> None:
    for card in data.get("cards", []):
        due = today + timedelta(days=card["due_in_days"])
        ledger.create(DEFAULT_OWNERS["srs_cards"], "srs_cards", front=card["front"], back=card["back"],
                      due_date=due.isoformat())
    for error in data.get("errors", []):
        ledger.create(DEFAULT_OWNERS["errors"], "errors", **error)
    for fact in data.get("facts", []):
        ledger.create(DEFAULT_OWNERS["facts"], "facts", **fact)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the demo database")
    parser.add_argument("--db", type=Path, default=DB_PATH, help=f"database to (re)create (default: {DB_PATH})")
    args = parser.parse_args(argv)
    seed(args.db)
    print(f"Demo database written to {args.db}")


if __name__ == "__main__":
    main()
