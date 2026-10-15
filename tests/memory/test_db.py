# Tests for the SQLite connection setup.

import operator
import sqlite3
from functools import partial

import pytest

from sensai.memory import db


class ConnectionOpener:

    def __init__(self):
        self.opened = []

    def __call__(self, path):
        connection = db.connect(path)
        self.opened.append(connection)
        return connection

    def close_all(self):
        for connection in self.opened:
            connection.close()


@pytest.fixture
def open_conn():
    opener = ConnectionOpener()
    yield opener
    opener.close_all()


def test_connect_memory_returns_row_factory_connection(open_conn):
    conn = open_conn(":memory:")
    assert isinstance(conn, sqlite3.Connection)
    assert conn.row_factory is sqlite3.Row


def test_connect_enables_foreign_keys(open_conn):
    conn = open_conn(":memory:")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_foreign_keys_are_enforced(open_conn):
    conn = open_conn(":memory:")
    conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, "
                 "parent_id INTEGER REFERENCES parent(id))")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO child (parent_id) VALUES (99)")


def test_connect_creates_parent_directory(tmp_path, open_conn):
    target = tmp_path / "nested" / "sensai.db"
    conn = open_conn(str(target))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    assert target.exists()


def test_connect_resolves_relative_path_via_core_config(tmp_path, monkeypatch, open_conn):
    monkeypatch.setattr(db, "resolve", partial(operator.truediv, tmp_path))
    conn = open_conn("nested/sensai.db")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    assert (tmp_path / "nested" / "sensai.db").exists()
