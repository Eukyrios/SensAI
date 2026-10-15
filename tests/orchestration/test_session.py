# Tests for Session: model switching, thinking, reset and resume.

import pytest

from sensai.llm.ollama import OllamaError
from sensai.orchestration.session import Session
from tests.fakes import FakeClient, make_config


@pytest.fixture
def session():
    return Session(make_config(), client=FakeClient())


def test_builds_real_client_and_default_agent():
    s = Session(make_config())
    assert s.client.url == "http://localhost:11434"
    assert s.spec.name == "sensei" and len(s.id) == 32 and s.history == []


def test_keeps_a_given_session_id():
    assert Session(make_config(), client=FakeClient(), session_id="abc").id == "abc"


def test_set_model_keeps_history(session):
    session.history.append({"role": "user", "content": "hello"})
    session.set_model("qwen3.5:4b")
    assert session.spec.model == "qwen3.5:4b"
    assert session.history == [{"role": "user", "content": "hello"}]


def test_set_model_rejects_missing(session):
    before = session.spec.model
    with pytest.raises(OllamaError):
        session.set_model("nope")
    assert session.spec.model == before


def test_reset(session):
    session.history.append({"role": "user", "content": "x"})
    session.reset()
    assert session.history == []


def test_set_thinking(session):
    session.set_thinking(True)
    assert session.spec.think is True
    session.set_thinking(False)
    assert session.spec.think is False


def test_set_thinking_rejected_when_model_cannot_reason():
    s = Session(make_config(), client=FakeClient(thinkers=()))
    with pytest.raises(ValueError, match="can't reason"):
        s.set_thinking(True)
    assert s.spec.think is False


def test_switching_to_a_model_that_cannot_reason_turns_thinking_off():
    s = Session(make_config(), client=FakeClient(thinkers=("qwen3.5:9b",)))
    s.set_thinking(True)
    s.set_model("qwen3.5:4b")
    assert (s.spec.model, s.spec.think) == ("qwen3.5:4b", False)


from sensai.memory.repository import SessionNotFound
from tests.conftest import memory_repository


def test_nothing_is_stored_until_the_first_message(session):
    assert session.repo.list_sessions() == []
    session.record("user", "hi")
    [row] = session.repo.list_sessions()
    assert row["id"] == session.id and row["model"] == session.spec.model


def test_record_chains_messages_on_the_active_branch(session):
    first = session.record("user", "hi")
    second = session.record("assistant", "hello")
    assert session.leaf == second
    assert session.repo.load_history(session.id) == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    session.discard(second, first)
    assert session.leaf == first
    assert [m["content"] for m in session.repo.load_history(session.id)] == ["hi"]


def test_resume_restores_history_and_branch():
    repo = memory_repository()
    first = Session(make_config(), client=FakeClient(), repo=repo)
    first.record("user", "hi")
    last = first.record("assistant", "hello")
    resumed = Session.resume(make_config(), first.id, client=FakeClient(), repo=repo)
    assert resumed.id == first.id and resumed.leaf == last and resumed.stored
    assert resumed.history == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_resume_unknown_session_raises():
    with pytest.raises(SessionNotFound):
        Session.resume(make_config(), "nope", client=FakeClient())


def test_reset_starts_a_new_session_and_keeps_the_old_one(session):
    old = session.id
    session.record("user", "hi")
    session.reset()
    assert session.id != old and session.leaf is None and not session.stored
    assert [m["content"] for m in session.repo.load_history(old)] == ["hi"]


def test_profile_comes_from_the_configured_name():
    repo = memory_repository()
    repo.save_profile("alice", "alice's profile")
    assert Session(make_config(), client=FakeClient(), repo=repo).profile is None
    s = Session(make_config(memory={"profile": "alice"}), client=FakeClient(), repo=repo)
    assert "alice's profile" in s.profile["content"]
