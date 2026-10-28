# Shared test fixtures: an in-memory repository so no test touches data/sensai.db.

import pytest

from sensai.memory import db
from sensai.memory.migrate import migrate
from sensai.memory.repository import SessionRepository

_opened: list[SessionRepository] = []


def memory_repository() -> SessionRepository:
    conn = db.connect(":memory:")
    migrate(conn)
    _opened.append(SessionRepository(conn))
    return _opened[-1]


def memory_repository_for(config: dict) -> SessionRepository:
    return memory_repository()


@pytest.fixture(autouse=True)
def no_external_processes(monkeypatch):
    """Tests that load config.toml never start its MCP servers nor reach the embedding model."""
    monkeypatch.setenv("SENSAI_MCP__ENABLED", "false")
    monkeypatch.setenv("SENSAI_RAG__ENABLED", "false")


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch):
    for module in ("sensai.orchestration.session", "sensai.surfaces.cli"):
        monkeypatch.setattr(f"{module}.open_repository", memory_repository_for)
    yield
    while _opened:
        _opened.pop().conn.close()
