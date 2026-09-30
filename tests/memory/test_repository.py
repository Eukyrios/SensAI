import sqlite3

import pytest

from sensai.memory.repository import open_repository
from tests.fakes import make_config


def test_create_session(repo):
    session_id = repo.create_session(model="qwen3.5:4b", title="Lesson 1")
    row = repo.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert (row["model"], row["title"]) == ("qwen3.5:4b", "Lesson 1")
    assert row["created_at"]


def test_create_session_without_title(repo):
    session_id = repo.create_session(model="qwen3.5:4b")
    row = repo.conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert row["title"] is None


def test_add_message_and_list_messages(repo):
    session_id = repo.create_session(model="qwen3.5:4b")
    repo.add_message(session_id, "user", "konnichiwa")
    repo.add_message(session_id, "assistant", "Konnichiwa!")
    messages = repo.list_messages(session_id)
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "konnichiwa"), ("assistant", "Konnichiwa!"),
    ]


def test_add_message_parent_id_defaults_to_null(repo):
    session_id = repo.create_session(model="m")
    message_id = repo.add_message(session_id, "user", "hello")
    row = repo.conn.execute("SELECT parent_id FROM messages WHERE id = ?", (message_id,)).fetchone()
    assert row["parent_id"] is None


def test_add_message_records_parent_id(repo):
    session_id = repo.create_session(model="m")
    original_id = repo.add_message(session_id, "assistant", "first answer")
    retry_id = repo.add_message(session_id, "assistant", "retried answer", parent_id=original_id)
    row = repo.conn.execute("SELECT parent_id FROM messages WHERE id = ?", (retry_id,)).fetchone()
    assert row["parent_id"] == original_id


def test_deleting_parent_message_cascades_to_children(repo):
    session_id = repo.create_session(model="m")
    original_id = repo.add_message(session_id, "assistant", "first answer")
    retry_id = repo.add_message(session_id, "assistant", "retried answer", parent_id=original_id)

    with repo.conn:
        repo.conn.execute("DELETE FROM messages WHERE id = ?", (original_id,))

    remaining_ids = {m["id"] for m in repo.list_messages(session_id)}
    assert retry_id not in remaining_ids


def test_record_agent_run_records_parent_run_id(repo):
    session_id = repo.create_session(model="m")
    user_id = repo.add_message(session_id, "user", "hello")
    assistant_id = repo.add_message(session_id, "assistant", "hi")
    turn_id = repo.record_turn(session_id, user_id, assistant_id)
    parent_run_id = repo.record_agent_run(turn_id, kind="rag", status="ok")
    child_run_id = repo.record_agent_run(turn_id, kind="tool", status="ok",
                                          parent_run_id=parent_run_id)
    row = repo.conn.execute(
        "SELECT parent_run_id FROM agent_runs WHERE id = ?", (child_run_id,)
    ).fetchone()
    assert row["parent_run_id"] == parent_run_id


def test_deleting_parent_agent_run_cascades_to_children(repo):
    session_id = repo.create_session(model="m")
    user_id = repo.add_message(session_id, "user", "hello")
    assistant_id = repo.add_message(session_id, "assistant", "hi")
    turn_id = repo.record_turn(session_id, user_id, assistant_id)
    parent_run_id = repo.record_agent_run(turn_id, kind="rag", status="ok")
    child_run_id = repo.record_agent_run(turn_id, kind="tool", status="ok",
                                          parent_run_id=parent_run_id)

    with repo.conn:
        repo.conn.execute("DELETE FROM agent_runs WHERE id = ?", (parent_run_id,))

    remaining = repo.conn.execute(
        "SELECT id FROM agent_runs WHERE id = ?", (child_run_id,)
    ).fetchall()
    assert remaining == []


def test_list_messages_is_scoped_to_session(repo):
    s1 = repo.create_session(model="m")
    s2 = repo.create_session(model="m")
    repo.add_message(s1, "user", "for s1")
    repo.add_message(s2, "user", "for s2")
    assert [m["content"] for m in repo.list_messages(s1)] == ["for s1"]


def test_invalid_role_rejected(repo):
    session_id = repo.create_session(model="qwen3.5:4b")
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_message(session_id, "narrator", "hello")


def test_record_turn(repo):
    session_id = repo.create_session(model="qwen3.5:4b")
    user_id = repo.add_message(session_id, "user", "hello")
    assistant_id = repo.add_message(session_id, "assistant", "hi")
    turn_id = repo.record_turn(session_id, user_id, assistant_id, latency_ms=120, tokens=42)
    row = repo.conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
    assert (row["latency_ms"], row["tokens"]) == (120, 42)


def test_record_agent_run(repo):
    session_id = repo.create_session(model="qwen3.5:4b")
    user_id = repo.add_message(session_id, "user", "hello")
    assistant_id = repo.add_message(session_id, "assistant", "hi")
    turn_id = repo.record_turn(session_id, user_id, assistant_id)
    run_id = repo.record_agent_run(turn_id, kind="guardrail", status="ok",
                                    input="hello", output="hello", duration_ms=5)
    row = repo.conn.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
    assert (row["kind"], row["status"], row["duration_ms"]) == ("guardrail", "ok", 5)


def test_cascade_delete_session_removes_messages_turns_and_runs(repo):
    session_id = repo.create_session(model="qwen3.5:4b")
    user_id = repo.add_message(session_id, "user", "hello")
    assistant_id = repo.add_message(session_id, "assistant", "hi")
    turn_id = repo.record_turn(session_id, user_id, assistant_id)
    repo.record_agent_run(turn_id, kind="tool", status="ok")

    with repo.conn:
        repo.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    assert repo.list_messages(session_id) == []
    assert repo.conn.execute(
        "SELECT * FROM turns WHERE session_id = ?", (session_id,)
    ).fetchall() == []
    assert repo.conn.execute(
        "SELECT * FROM agent_runs WHERE turn_id = ?", (turn_id,)
    ).fetchall() == []


def test_open_repository_wires_config_to_a_migrated_db():
    repo = open_repository(make_config(storage={"db_path": ":memory:"}))
    try:
        session_id = repo.create_session(model="m")
        assert repo.list_messages(session_id) == []
    finally:
        repo.conn.close()
