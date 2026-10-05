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


def test_generated_ids_are_uuid_hex(repo):
    session_id = repo.create_session(model="m")
    message_id = repo.add_message(session_id, "user", "hello")
    for value in (session_id, message_id):
        assert isinstance(value, str) and len(value) == 32
        int(value, 16)      # hex
    assert session_id != message_id


def test_writes_accept_the_ids_the_events_already_carry(repo):
    from sensai.core.ids import new_id
    session_id, user_id, turn_id, run_id, child_id = (new_id() for _ in range(5))
    assert repo.create_session(model="m", session_id=session_id) == session_id
    assert repo.add_message(session_id, "user", "hello", message_id=user_id) == user_id
    assert repo.record_turn(session_id, user_id, turn_id=turn_id) == turn_id
    assert repo.record_agent_run(turn_id, kind="sensei", status="ok", run_id=run_id) == run_id
    assert repo.record_agent_run(turn_id, kind="tool", status="ok",
                                 parent_run_id=run_id, run_id=child_id) == child_id
    row = repo.conn.execute("SELECT turn_id, parent_run_id FROM agent_runs WHERE id = ?",
                            (child_id,)).fetchone()
    assert (row["turn_id"], row["parent_run_id"]) == (turn_id, run_id)


def test_failed_turn_is_recorded_without_assistant_message(repo):
    session_id = repo.create_session(model="m")
    user_id = repo.add_message(session_id, "user", "hello")
    turn_id = repo.record_turn(session_id, user_id)
    row = repo.conn.execute("SELECT assistant_message_id FROM turns WHERE id = ?", (turn_id,)).fetchone()
    assert row["assistant_message_id"] is None


def test_list_messages_keeps_insertion_order_despite_random_ids(repo):
    session_id = repo.create_session(model="m")
    ids = ["ffff", "0000", "8888", "1111"]         # deliberately not sorted
    for i, message_id in enumerate(ids):
        repo.add_message(session_id, "user", f"m{i}", message_id=message_id)
    assert [m["id"] for m in repo.list_messages(session_id)] == ids


def test_duplicate_id_rejected(repo):
    session_id = repo.create_session(model="m")
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_session(model="m", session_id=session_id)


# --- session persistence (#67, #68) ---

from sensai.memory.repository import SessionNotFound


def _chat(repo, session_id, *texts):
    """Add alternating user/assistant messages as one chain; return their ids."""
    ids, parent = [], None
    for i, text in enumerate(texts):
        parent = repo.add_message(session_id, "user" if i % 2 == 0 else "assistant", text, parent)
        ids.append(parent)
    return ids


def test_add_message_moves_active_leaf(repo):
    session_id = repo.create_session(model="m")
    first, second = _chat(repo, session_id, "hi", "hello")
    assert repo.get_session(session_id)["active_leaf_id"] == second
    assert first != second


def test_save_then_reload_gives_same_history(repo):
    session_id = repo.create_session(model="m")
    _chat(repo, session_id, "hi", "hello", "how are you", "fine")
    assert repo.load_history(session_id) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you"},
        {"role": "assistant", "content": "fine"},
    ]


def test_load_history_of_empty_session(repo):
    assert repo.load_history(repo.create_session(model="m")) == []


def test_reload_follows_active_leaf_across_branches(repo):
    session_id = repo.create_session(model="m")
    hi, hello = _chat(repo, session_id, "hi", "hello")
    other = repo.add_message(session_id, "assistant", "hey!", parent_id=hi)   # sibling branch
    assert [m["content"] for m in repo.load_history(session_id)] == ["hi", "hey!"]
    repo.set_active_leaf(session_id, hello)
    assert [m["content"] for m in repo.load_history(session_id)] == ["hi", "hello"]
    repo.set_active_leaf(session_id, other)
    assert [m["content"] for m in repo.load_history(session_id)] == ["hi", "hey!"]


def test_set_active_leaf_rejects_message_of_another_session(repo):
    a, b = repo.create_session(model="m"), repo.create_session(model="m")
    foreign = repo.add_message(b, "user", "x")
    own = repo.add_message(a, "user", "y")
    with pytest.raises(SessionNotFound):
        repo.set_active_leaf(a, foreign)
    assert repo.get_session(a)["active_leaf_id"] == own


def test_unknown_session_raises(repo):
    for call in (repo.get_session, repo.load_history):
        with pytest.raises(SessionNotFound):
            call("nope")
    with pytest.raises(SessionNotFound):
        repo.set_active_leaf("nope", "x")


def test_list_sessions_most_recent_first(repo):
    first, second = repo.create_session(model="m"), repo.create_session(model="m")
    assert [s["id"] for s in repo.list_sessions()] == [second, first]


def test_discard_leaf_steps_back_to_parent(repo):
    session_id = repo.create_session(model="m")
    hi, hello = _chat(repo, session_id, "hi", "hello")
    repo.discard_leaf(hello)
    assert repo.get_session(session_id)["active_leaf_id"] == hi
    assert [m["content"] for m in repo.load_history(session_id)] == ["hi"]
    repo.discard_leaf(hi)
    assert repo.get_session(session_id)["active_leaf_id"] is None


# --- learner profile (#69) ---

def test_profile_roundtrip_and_update(repo):
    assert repo.get_profile() is None
    repo.save_profile("default", "N5 learner")
    assert repo.get_profile("default") == "N5 learner"
    repo.save_profile("default", "N4 learner")
    assert repo.get_profile("default") == "N4 learner"
    assert repo.get_profile("other") is None


def test_profile_system_message(repo):
    assert repo.profile_system_message() is None
    repo.save_profile("default", "N5 learner")
    msg = repo.profile_system_message()
    assert msg["role"] == "system" and "N5 learner" in msg["content"]
    assert repo.profile_system_message("other") is None
