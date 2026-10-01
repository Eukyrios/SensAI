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
