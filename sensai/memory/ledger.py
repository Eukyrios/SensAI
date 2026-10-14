# Learner ledger: errors, srs_cards and facts, written by agents, one owner per table.

from __future__ import annotations

import sqlite3

from ..core.ids import new_id
from ..knowledge.tools import Tool

COLUMNS = {
    "errors": ("session_id", "sentence", "correction", "error_type", "resolved"),
    "srs_cards": ("front", "back", "ease", "interval_days", "due_date"),
    "facts": ("key", "value"),
}
REQUIRED = {
    "errors": ("sentence", "correction", "error_type"),
    "srs_cards": ("front", "back"),
    "facts": ("key", "value"),
}
DEFAULT_OWNERS = {"errors": "corrector", "srs_cards": "sensei", "facts": "sensei"}


class WriteNotAllowed(PermissionError):
    pass


class RowNotFound(LookupError):
    pass


class Ledger:
    def __init__(self, conn: sqlite3.Connection, owners: dict[str, str] | None = None):
        self.conn = conn
        self.owners = {**DEFAULT_OWNERS, **(owners or {})}

    def _check_table(self, table: str) -> None:
        if table not in COLUMNS:
            raise ValueError(f"unknown table: {table} (expected one of {', '.join(COLUMNS)})")

    def _check_write(self, agent: str, table: str) -> None:
        self._check_table(table)
        if self.owners[table] != agent:
            raise WriteNotAllowed(f"{agent} cannot write {table}: it belongs to {self.owners[table]}")

    def _check_fields(self, table: str, fields: dict) -> None:
        unknown = set(fields) - set(COLUMNS[table])
        if unknown:
            raise ValueError(f"unknown field(s) for {table}: {', '.join(sorted(unknown))}")

    def create(self, agent: str, table: str, *, row_id: str | None = None, **fields) -> str:
        self._check_write(agent, table)
        self._check_fields(table, fields)
        missing = [column for column in REQUIRED[table] if column not in fields]
        if missing:
            raise ValueError(f"missing field(s) for {table}: {', '.join(missing)}")
        row_id = row_id or new_id()
        columns = ", ".join(("id", *fields))
        placeholders = ", ".join("?" * (len(fields) + 1))
        with self.conn:
            self.conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                              (row_id, *fields.values()))
        return row_id

    def read(self, table: str, row_id: str) -> sqlite3.Row:
        self._check_table(table)
        row = self.conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise RowNotFound(f"{table} {row_id}")
        return row

    def list(self, table: str, **filters) -> list[sqlite3.Row]:
        self._check_table(table)
        self._check_fields(table, filters)
        where = " AND ".join(f"{column} = ?" for column in filters)
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + " ORDER BY rowid"
        return self.conn.execute(sql, tuple(filters.values())).fetchall()

    def update(self, agent: str, table: str, row_id: str, **fields) -> None:
        self._check_write(agent, table)
        self._check_fields(table, fields)
        if not fields:
            raise ValueError("nothing to update")
        assignments = ", ".join(f"{column} = ?" for column in fields)
        if table == "facts":
            assignments += ", updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        with self.conn:
            cursor = self.conn.execute(f"UPDATE {table} SET {assignments} WHERE id = ?",
                                       (*fields.values(), row_id))
        if cursor.rowcount == 0:
            raise RowNotFound(f"{table} {row_id}")

    def delete(self, agent: str, table: str, row_id: str) -> None:
        self._check_write(agent, table)
        with self.conn:
            cur = self.conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        if cur.rowcount == 0:
            raise RowNotFound(f"{table} {row_id}")

    def tools(self, agent: str) -> list[Tool]:
        return _LedgerToolset(self, agent).tools()


_TABLE_ARGUMENT = {"type": "string", "enum": list(COLUMNS)}
_FIELDS_ARGUMENT = {"type": "object", "description": "column name -> value"}
_ID_ARGUMENT = {"type": "string"}


def _object_schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


class _LedgerToolset:

    def __init__(self, ledger: Ledger, agent: str):
        self.ledger = ledger
        self.agent = agent

    def tools(self) -> list[Tool]:
        return [
            Tool("ledger_create", "Add a row to errors, srs_cards or facts (only the tables you own).",
                 _object_schema({"table": _TABLE_ARGUMENT, "fields": _FIELDS_ARGUMENT}, ["table", "fields"]),
                 self.create),
            Tool("ledger_read", "Read one row by id, or list rows of errors, srs_cards or facts.",
                 _object_schema({"table": _TABLE_ARGUMENT, "id": _ID_ARGUMENT, "filters": _FIELDS_ARGUMENT},
                                ["table"]),
                 self.read),
            Tool("ledger_update", "Change fields of a row you own.",
                 _object_schema({"table": _TABLE_ARGUMENT, "id": _ID_ARGUMENT, "fields": _FIELDS_ARGUMENT},
                                ["table", "id", "fields"]),
                 self.update),
            Tool("ledger_delete", "Delete a row you own.",
                 _object_schema({"table": _TABLE_ARGUMENT, "id": _ID_ARGUMENT}, ["table", "id"]),
                 self.delete),
        ]

    def create(self, table: str, fields: dict) -> dict:
        return {"id": self.ledger.create(self.agent, table, **fields)}

    def read(self, table: str, id: str | None = None, filters: dict | None = None):
        if id is not None:
            return dict(self.ledger.read(table, id))
        return [dict(row) for row in self.ledger.list(table, **(filters or {}))]

    def update(self, table: str, id: str, fields: dict) -> dict:
        self.ledger.update(self.agent, table, id, **fields)
        return {"updated": id}

    def delete(self, table: str, id: str) -> dict:
        self.ledger.delete(self.agent, table, id)
        return {"deleted": id}
