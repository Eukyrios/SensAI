# Tests for the demo database: a fresh database holding the seeded profile and sessions.

from scripts import seed_demo
from sensai.memory import db
from sensai.memory.repository import SessionRepository

SEED = '''
profile = "Alex, a beginner."

[[sessions]]
title = "Greetings"
messages = [
    { role = "user", content = "How do I say hello?" },
    { role = "assistant", content = "こんにちは (konnichiwa) — hello." },
]
'''


def write_seed(tmp_path):
    seed_path = tmp_path / "seed.toml"
    seed_path.write_text(SEED, encoding="utf-8")
    return seed_path


def open_seeded(db_path):
    return SessionRepository(db.connect(str(db_path)))


def test_seed_stores_the_profile(tmp_path):
    seed_demo.seed(tmp_path / "demo.db", write_seed(tmp_path)).conn.close()
    repo = open_seeded(tmp_path / "demo.db")
    assert repo.get_profile("default") == "Alex, a beginner."


def test_seed_stores_each_session_as_a_resumable_chain(tmp_path):
    seed_demo.seed(tmp_path / "demo.db", write_seed(tmp_path)).conn.close()
    repo = open_seeded(tmp_path / "demo.db")
    (session,) = repo.list_sessions()
    assert session["title"] == "Greetings"
    assert repo.load_history(session["id"]) == [
        {"role": "user", "content": "How do I say hello?"},
        {"role": "assistant", "content": "こんにちは (konnichiwa) — hello."},
    ]


def test_seed_starts_from_an_empty_database(tmp_path):
    seed_path = write_seed(tmp_path)
    seed_demo.seed(tmp_path / "demo.db", seed_path).conn.close()
    seed_demo.seed(tmp_path / "demo.db", seed_path).conn.close()
    assert len(open_seeded(tmp_path / "demo.db").list_sessions()) == 1


def test_the_shipped_seed_loads(tmp_path):
    repo = seed_demo.seed(tmp_path / "demo.db")
    assert repo.list_sessions()
    repo.conn.close()


def test_main_writes_the_given_database(tmp_path, capsys):
    seed_demo.main(["--db", str(tmp_path / "demo.db")])
    assert "demo.db" in capsys.readouterr().out
    assert (tmp_path / "demo.db").exists()
