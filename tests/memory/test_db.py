# Tests for the SQLite connection setup.

import sqlite3

import pytest

from sensai.memory import db


@pytest.fixture
def open_conn():
    opened = []

    def _connect(path):
        c = db.connect(path)
        opened.append(c)
        return c

    yield _connect
    for c in opened:
        c.close()


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
    monkeypatch.setattr(db, "resolve", lambda p: tmp_path / p)
    conn = open_conn("nested/sensai.db")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    assert (tmp_path / "nested" / "sensai.db").exists()
