import pytest

from sensai.chat import Agent, DEFAULT_SYSTEM
from tests.conftest import FakeClient


@pytest.fixture
def agent(config):
    a = Agent(config)
    a.client = FakeClient()
    return a


def test_model_override(config):
    assert Agent(config, model="autre").client.model == "autre"
    assert Agent(config).client.model == config["ollama"]["model"]


def test_chat_streams_and_records_history(agent):
    assert "".join(agent.chat("  salut  ")) == "Bonjour"
    assert agent.history == [
        {"role": "user", "content": "salut"},
        {"role": "assistant", "content": "Bonjour"},
    ]
    assert agent.client.calls[0][0] == {"role": "system", "content": DEFAULT_SYSTEM}


def test_empty_message_rejected(agent):
    with pytest.raises(ValueError):
        list(agent.chat("   "))


def test_failure_removes_user_message(agent):
    agent.client = FakeClient(error=RuntimeError("down"))
    with pytest.raises(RuntimeError):
        list(agent.chat("salut"))
    assert agent.history == []


def test_history_is_truncated(agent):
    agent.config["memory"]["max_history_turns"] = 1
    for i in range(3):
        list(agent.chat(f"m{i}"))
    sent = agent.client.calls[-1]
    assert [m["content"] for m in sent[1:]] == ["Bonjour", "m2"]


def test_persona_confirm_reset(agent):
    agent.set_persona("casual")
    assert agent.persona == "casual"
    assert agent.pending_action() is None
    agent._pending = {"tool": "send_email", "args": {}, "summary": "s"}
    assert agent.confirm(True) == "Action confirmée."
    assert agent.pending_action() is None
    assert agent.confirm(False) == "Action annulée."
    list(agent.chat("x"))
    agent.reset()
    assert agent.history == []
