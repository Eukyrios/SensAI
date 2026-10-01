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
