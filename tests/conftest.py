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


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch):
    for module in ("sensai.orchestration.session", "sensai.surfaces.cli"):
        monkeypatch.setattr(f"{module}.open_repository", lambda config: memory_repository())
    yield
    while _opened:
        _opened.pop().conn.close()
