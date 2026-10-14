# Tests for the async repository facade.

from sensai.memory.repository import AsyncSessionRepository, open_async_repository
from tests.fakes import make_config


async def test_async_full_round_trip(async_repo):
    session_id = await async_repo.create_session(model="qwen3.5:4b", title="Async lesson")
    user_id = await async_repo.add_message(session_id, "user", "hello")
    assistant_id = await async_repo.add_message(session_id, "assistant", "hi")
    turn_id = await async_repo.record_turn(session_id, user_id, assistant_id,
                                            latency_ms=50, tokens=10)
    await async_repo.record_agent_run(turn_id, kind="rag", status="ok")

    messages = await async_repo.list_messages(session_id)
    assert [m["content"] for m in messages] == ["hello", "hi"]


async def test_async_repository_wraps_the_sync_one(repo):
    async_repo = AsyncSessionRepository(repo)
    session_id = await async_repo.create_session(model="m")
    assert repo.conn.execute(
        "SELECT model FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()["model"] == "m"


async def test_open_async_repository_wires_config_to_a_migrated_db():
    async_repo = open_async_repository(make_config(storage={"db_path": ":memory:"}))
    try:
        session_id = await async_repo.create_session(model="m")
        assert await async_repo.list_messages(session_id) == []
    finally:
        async_repo.repo.conn.close()


async def test_async_persistence_and_profile(async_repo):
    session_id = await async_repo.create_session("m")
    hi = await async_repo.add_message(session_id, "user", "hi")
    hello = await async_repo.add_message(session_id, "assistant", "hello", parent_id=hi)
    assert [m["content"] for m in await async_repo.load_history(session_id)] == ["hi", "hello"]
    assert (await async_repo.get_session(session_id))["active_leaf_id"] == hello
    assert len(await async_repo.list_sessions()) == 1
    await async_repo.set_active_leaf(session_id, hi)
    await async_repo.discard_leaf(hello)
    assert await async_repo.get_profile() is None
    await async_repo.save_profile("default", "N5")
    assert await async_repo.get_profile() == "N5"
    assert (await async_repo.profile_system_message())["role"] == "system"
