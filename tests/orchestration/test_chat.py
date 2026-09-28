import pytest

from sensai.orchestration.chat import Chat
from tests.fakes import FakeClient, make_config


@pytest.fixture
def chat():
    return Chat(make_config(), client=FakeClient())


def test_builds_real_client_by_default():
    assert Chat(make_config()).client.url == "http://localhost:11434"


def test_streams_and_records_history(chat):
    assert "".join(chat.send("  hello  ")) == "Konnichiwa"
    assert chat.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Konnichiwa"},
    ]
    model, messages, options = chat.client.calls[0]
    assert model == chat.spec.model and chat.spec.name == "sensei"
    assert options == {"num_ctx": chat.spec.num_ctx, "temperature": chat.spec.temperature,
                       "think": chat.spec.think}
    assert messages[0] == {"role": "system", "content": chat.spec.system_prompt}


def test_empty_message_rejected(chat):
    with pytest.raises(ValueError):
        list(chat.send("   "))


def test_failure_drops_user_message(chat):
    chat.client = FakeClient(error=RuntimeError("down"))
    with pytest.raises(RuntimeError):
        list(chat.send("hello"))
    assert chat.history == []


def test_history_is_truncated():
    chat = Chat(make_config(memory={"max_history_turns": 1}), client=FakeClient())
    for i in range(3):
        list(chat.send(f"m{i}"))
    assert [m["content"] for m in chat.client.calls[-1][1][1:]] == ["Konnichiwa", "m2"]


def test_reset(chat):
    list(chat.send("x"))
    chat.reset()
    assert chat.history == []


def test_set_model_keeps_history(chat):
    list(chat.send("hello"))
    chat.set_model("qwen3.5:4b")
    list(chat.send("again"))
    model, messages, _ = chat.client.calls[-1]
    assert model == "qwen3.5:4b"
    assert [m["content"] for m in messages[1:]] == ["hello", "Konnichiwa", "again"]


def test_set_model_rejects_missing(chat):
    from sensai.llm.ollama import OllamaError
    before = chat.spec.model
    with pytest.raises(OllamaError):
        chat.set_model("nope")
    assert chat.spec.model == before
