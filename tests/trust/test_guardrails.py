from sensai import guardrails, rag, tools
from sensai.config import load_config, CONFIG_PATH


def test_load_config():
    cfg = load_config(CONFIG_PATH)
    assert "ollama" in cfg


def test_guardrails_passthrough(config):
    assert guardrails.filter_input("a", config) == "a"
    assert guardrails.filter_output("b", config) == "b"


def test_rag_stub():
    assert rag.ingest("data/mails") == 0
    chunks = rag.retrieve("réunion", k=1)
    assert len(chunks) == 1 and chunks[0]["id"] == "stub-1"
    assert rag.retrieve("x", k=0) == []


def test_tools(config):
    assert "send_email" in tools.run_tool("send_email", {})
    assert tools.needs_confirmation("send_email", config)
    assert not tools.needs_confirmation("search", config)
    assert not tools.needs_confirmation("send_email", {})
