# Tests for the SQLite connection setup.

import operator
import sqlite3
import threading
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


THREADS = 8
WRITES_PER_THREAD = 50


def run_threads(target, count):
    errors = []

    def guarded(index):
        try:
            target(index)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=guarded, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_concurrent_writes_from_many_threads_are_all_stored(open_conn):
    conn = open_conn(":memory:")
    conn.execute("CREATE TABLE t (thread INTEGER, n INTEGER)")

    def write(index):
        for n in range(WRITES_PER_THREAD):
            with conn:
                conn.execute("INSERT INTO t (thread, n) VALUES (?, ?)", (index, n))

    assert run_threads(write, THREADS) == []
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == THREADS * WRITES_PER_THREAD


def test_transaction_is_not_interleaved_by_another_thread(open_conn):
    conn = open_conn(":memory:")
    conn.execute("CREATE TABLE t (thread INTEGER, n INTEGER)")

    def write(index):
        for n in range(WRITES_PER_THREAD):
            with conn:
                conn.execute("INSERT INTO t (thread, n) VALUES (?, ?)", (index, 0))
                conn.execute("INSERT INTO t (thread, n) VALUES (?, ?)", (index, 1))

    assert run_threads(write, THREADS) == []
    rows = conn.execute("SELECT thread, n FROM t ORDER BY rowid").fetchall()
    assert len(rows) == THREADS * WRITES_PER_THREAD * 2
    for first, second in zip(rows[0::2], rows[1::2]):
        assert (first["thread"], first["n"], second["n"]) == (second["thread"], 0, 1)


def test_rolled_back_transaction_does_not_leak_to_other_threads(open_conn):
    conn = open_conn(":memory:")
    conn.execute("CREATE TABLE t (thread INTEGER)")

    def write(index):
        for _ in range(WRITES_PER_THREAD):
            try:
                with conn:
                    conn.execute("INSERT INTO t (thread) VALUES (?)", (index,))
                    if index % 2:
                        raise ValueError("abort")
            except ValueError:
                pass

    assert run_threads(write, THREADS) == []
    odd = conn.execute("SELECT COUNT(*) FROM t WHERE thread % 2 = 1").fetchone()[0]
    even = conn.execute("SELECT COUNT(*) FROM t WHERE thread % 2 = 0").fetchone()[0]
    assert (odd, even) == (0, THREADS // 2 * WRITES_PER_THREAD)


def test_repository_is_safe_with_concurrent_add_message(repo):
    session_id = repo.create_session("m")

    def write(index):
        for n in range(WRITES_PER_THREAD):
            repo.add_message(session_id, "user", f"{index}-{n}")

    assert run_threads(write, THREADS) == []
    assert len(repo.list_messages(session_id)) == THREADS * WRITES_PER_THREAD


def test_locked_cursor_and_connection_helpers(open_conn):
    conn = open_conn(":memory:")
    conn.executescript("CREATE TABLE t (n INTEGER);")
    conn.executemany("INSERT INTO t (n) VALUES (?)", [(1,), (2,), (3,)])
    cursor = conn.execute("SELECT n FROM t ORDER BY n")
    assert [row["n"] for row in cursor.fetchmany(2)] == [1, 2]
    assert cursor.fetchone()["n"] == 3
    assert conn.cursor().executemany("INSERT INTO t (n) VALUES (?)", [(4,)]).rowcount == 1

