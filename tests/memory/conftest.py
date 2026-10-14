# Fixtures for the storage tests.

import pytest

from sensai.memory import db
from sensai.memory.migrate import migrate
from sensai.memory.repository import AsyncSessionRepository, SessionRepository


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    migrate(connection)
    yield connection
    connection.close()


@pytest.fixture
def repo(conn):
    return SessionRepository(conn)


@pytest.fixture
def async_repo(repo):
    return AsyncSessionRepository(repo)
