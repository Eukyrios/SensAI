from sensai.memory import db
from sensai.memory.migrate import MIGRATIONS_DIR, migrate

LATEST = len(list(MIGRATIONS_DIR.glob("*.sql")))


def test_migrate_creates_all_tables(conn):
    tables = {row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert {"sessions", "messages", "turns", "agent_runs", "learner_profiles"} <= tables


def test_sessions_have_active_leaf_column(conn):
    assert "active_leaf_id" in {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}


def test_messages_and_agent_runs_have_self_referencing_parent_columns(conn):
    message_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(agent_runs)")}
    assert "parent_id" in message_cols
    assert "parent_run_id" in run_cols


def test_migrate_creates_indexes(conn):
    indexes = {row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    )}
    assert {"idx_messages_session_id", "idx_turns_session_id", "idx_agent_runs_turn_id"} <= indexes


def test_migrate_sets_user_version(conn):
    assert conn.execute("PRAGMA user_version").fetchone()[0] == LATEST


def test_migrate_is_idempotent(conn):
    migrate(conn)  # rerunning must not raise (e.g. "table already exists")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == LATEST


def test_migrate_skips_already_applied_versions():
    conn = db.connect(":memory:")
    try:
        conn.execute(f"PRAGMA user_version = {LATEST}")
        migrate(conn)  # nothing pending: must not try to recreate the tables
        tables = {row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert tables == set()
    finally:
        conn.close()


def test_ids_are_text_primary_keys(conn):
    for table in ("sessions", "messages", "turns", "agent_runs"):
        cols = {row["name"]: row for row in conn.execute(f"PRAGMA table_info({table})")}
        assert cols["id"]["type"] == "TEXT" and cols["id"]["pk"] == 1
        for name, col in cols.items():
            if name.endswith("_id"):
                assert col["type"] == "TEXT", f"{table}.{name}"
