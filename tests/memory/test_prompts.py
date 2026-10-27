# Tests for the prompt store: prompts kept by content hash, pins and metrics grouped by hash.

import pytest

from sensai.core.agent_spec import AgentSpec, SpecError, prompt_hash
from sensai.memory.prompts import PromptStore


@pytest.fixture
def store(conn):
    return PromptStore(conn)


def make_spec(system_prompt="Be kind.", prompt_version=""):
    return AgentSpec(name="sensei", model="m", num_ctx=2048, system_prompt=system_prompt,
                     prompt_version=prompt_version)


def record_run(repo, prompt, status="ok", duration_ms=100, tokens=50, agent="sensei"):
    session_id = repo.create_session(model="m")
    user_id = repo.add_message(session_id, "user", "hello")
    turn_id = repo.record_turn(session_id, user_id)
    repo.record_agent_run(turn_id, "agent", status, duration_ms=duration_ms, agent=agent,
                          prompt_hash=prompt, tokens=tokens)


def test_prompt_hash_is_short_and_stable():
    assert prompt_hash("Be kind.") == prompt_hash("Be kind.")
    assert len(prompt_hash("Be kind.")) == 12


def test_editing_a_prompt_changes_its_hash():
    assert prompt_hash("Be kind.") != prompt_hash("Be kind!")
    assert make_spec("Be kind.").prompt_hash == prompt_hash("Be kind.")


def test_save_stores_the_prompt_under_its_hash(store):
    content_hash = store.save("sensei", "Be kind.")
    assert content_hash == prompt_hash("Be kind.")
    assert store.get("sensei", content_hash) == "Be kind."


def test_saving_the_same_prompt_twice_keeps_one_version(store):
    store.save("sensei", "Be kind.")
    store.save("sensei", "Be kind.")
    assert len(store.versions("sensei")) == 1


def test_versions_lists_each_prompt_of_an_agent_oldest_first(store):
    store.save("sensei", "Be kind.")
    store.save("sensei", "Be kind and brief.")
    store.save("checker", "Check the sentence.")
    hashes = [row["hash"] for row in store.versions("sensei")]
    assert hashes == [prompt_hash("Be kind."), prompt_hash("Be kind and brief.")]


def test_resolve_saves_an_unpinned_prompt(store):
    spec = make_spec("Be kind.")
    assert store.resolve(spec) is spec
    assert store.get("sensei", spec.prompt_hash) == "Be kind."


def test_resolve_swaps_in_the_pinned_prompt(store):
    old_hash = store.save("sensei", "Be kind.")
    pinned = store.resolve(make_spec("Be kind and brief.", prompt_version=old_hash))
    assert pinned.system_prompt == "Be kind." and pinned.prompt_hash == old_hash


def test_resolve_does_not_save_the_prompt_a_pin_replaces(store):
    old_hash = store.save("sensei", "Be kind.")
    store.resolve(make_spec("Be kind and brief.", prompt_version=old_hash))
    assert [row["hash"] for row in store.versions("sensei")] == [old_hash]


def test_unknown_pin_is_a_spec_error(store):
    with pytest.raises(SpecError, match="unknown prompt version 000000000000"):
        store.resolve(make_spec(prompt_version="000000000000"))


def test_metrics_are_grouped_by_prompt_hash(store, repo):
    old_hash = store.save("sensei", "Be kind.")
    new_hash = store.save("sensei", "Be kind and brief.")
    record_run(repo, old_hash, duration_ms=100, tokens=40)
    record_run(repo, old_hash, status="timeout", duration_ms=300, tokens=60)
    record_run(repo, new_hash, duration_ms=50, tokens=20)
    rows = [tuple(row) for row in store.metrics("sensei")]
    assert rows == [(old_hash, 2, 1, 200.0, 50.0), (new_hash, 1, 1, 50.0, 20.0)]


def test_metrics_leave_out_other_agents_and_runs_without_a_prompt(store, repo):
    record_run(repo, store.save("checker", "Check."), agent="checker")
    record_run(repo, None)
    assert store.metrics("sensei") == []
