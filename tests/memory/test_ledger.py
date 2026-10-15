# Tests for the learner ledger and its per-agent write ownership.

import json

import pytest

from sensai.memory.ledger import Ledger, RowNotFound, WriteNotAllowed
from sensai.knowledge.tools import ToolRegistry


@pytest.fixture
def ledger(conn):
    return Ledger(conn)


def test_migration_creates_ledger_tables(conn):
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"errors", "srs_cards", "facts"} <= tables


def test_error_crud(ledger):
    eid = ledger.create("corrector", "errors", sentence="私は学校に行きます", correction="私は学校へ行きます",
                        error_type="particle")
    row = ledger.read("errors", eid)
    assert (row["error_type"], row["resolved"]) == ("particle", 0)
    ledger.update("corrector", "errors", eid, resolved=1)
    assert ledger.read("errors", eid)["resolved"] == 1
    ledger.delete("corrector", "errors", eid)
    with pytest.raises(RowNotFound):
        ledger.read("errors", eid)


def test_srs_card_defaults_and_list_filter(ledger):
    cid = ledger.create("sensei", "srs_cards", front="猫", back="cat")
    row = ledger.read("srs_cards", cid)
    assert (row["ease"], row["interval_days"]) == (2.5, 0)
    assert row["due_date"]
    ledger.create("sensei", "srs_cards", front="犬", back="dog")
    assert [r["id"] for r in ledger.list("srs_cards", front="猫")] == [cid]
    assert len(ledger.list("srs_cards")) == 2


def test_fact_update_touches_updated_at(ledger, conn):
    fid = ledger.create("sensei", "facts", key="level", value="N5")
    conn.execute("UPDATE facts SET updated_at = '2000-01-01' WHERE id = ?", (fid,))
    ledger.update("sensei", "facts", fid, value="N4")
    row = ledger.read("facts", fid)
    assert row["value"] == "N4" and row["updated_at"] > "2000-01-01"


def test_only_the_owner_writes(ledger):
    with pytest.raises(WriteNotAllowed):
        ledger.create("sensei", "errors", sentence="a", correction="b", error_type="c")
    eid = ledger.create("corrector", "errors", sentence="a", correction="b", error_type="c")
    with pytest.raises(WriteNotAllowed):
        ledger.update("sensei", "errors", eid, resolved=1)
    with pytest.raises(WriteNotAllowed):
        ledger.delete("sensei", "errors", eid)
    assert ledger.read("errors", eid)["resolved"] == 0


def test_owners_can_be_overridden(conn):
    ledger = Ledger(conn, owners={"facts": "profiler"})
    ledger.create("profiler", "facts", key="k", value="v")
    with pytest.raises(WriteNotAllowed):
        ledger.create("sensei", "facts", key="k", value="v")


def test_rejects_bad_input(ledger):
    with pytest.raises(ValueError, match="unknown table"):
        ledger.list("sessions")
    with pytest.raises(ValueError, match="unknown field"):
        ledger.create("sensei", "facts", key="k", value="v", id="forced")
    with pytest.raises(ValueError, match="missing field"):
        ledger.create("sensei", "facts", key="k")
    with pytest.raises(RowNotFound):
        ledger.update("sensei", "facts", "nope", value="v")


def test_error_session_is_kept_nullable_on_session_delete(ledger, repo, conn):
    sid = repo.create_session("m")
    eid = ledger.create("corrector", "errors", session_id=sid, sentence="a", correction="b", error_type="c")
    with conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    assert ledger.read("errors", eid)["session_id"] is None


def run(registry, name, args):
    return registry.execute(name, args)


def test_tools_drive_crud_for_the_owning_agent(ledger):
    registry = ToolRegistry()
    for tool in ledger.tools("corrector"):
        registry.register(tool)
    created = json.loads(run(registry, "ledger_create", {
        "table": "errors", "fields": {"sentence": "a", "correction": "b", "error_type": "c"}}))
    eid = created["id"]
    assert json.loads(run(registry, "ledger_read", {"table": "errors", "id": eid}))["sentence"] == "a"
    assert len(json.loads(run(registry, "ledger_read", {"table": "errors", "filters": {"resolved": 0}}))) == 1
    run(registry, "ledger_update", {"table": "errors", "id": eid, "fields": {"resolved": 1}})
    assert ledger.read("errors", eid)["resolved"] == 1
    run(registry, "ledger_delete", {"table": "errors", "id": eid})
    assert ledger.list("errors") == []


def test_tools_refuse_a_table_the_agent_does_not_own(ledger):
    registry = ToolRegistry()
    for tool in ledger.tools("sensei"):
        registry.register(tool)
    result = run(registry, "ledger_create", {
        "table": "errors", "fields": {"sentence": "a", "correction": "b", "error_type": "c"}})
    assert result.startswith("Error executing ledger_create") and "belongs to corrector" in result
    assert ledger.list("errors") == []
