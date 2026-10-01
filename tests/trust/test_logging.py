"""Tests for the JSONL logger."""
import json
import pytest
from pathlib import Path

from sensai.core.events import Token, TurnFinished, TurnStarted
from sensai.trust.logging import JsonlLogger


@pytest.mark.asyncio
async def test_jsonl_logger_writes_events(tmp_path: Path):
    log_file = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_file)

    async def mock_stream():
        yield TurnStarted(session_id="s1", turn_id="t1", user_input="Hello")
        yield Token(turn_id="t1", run_id="r1", text="Hello")
        yield TurnFinished(turn_id="t1", model="qwen3.5:9b")

    events = []
    async for event in logger.log_stream(mock_stream()):
        events.append(event)

    assert len(events) == 3
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    first_record = json.loads(lines[0])
    assert first_record["event_type"] == "TurnStarted"
    assert first_record["session_id"] == "s1"
    assert first_record["user_input"] == "Hello"

    second_record = json.loads(lines[1])
    assert second_record["event_type"] == "Token"
    assert second_record["text"] == "Hello"
    assert second_record["turn_id"] == "t1"
    assert second_record["run_id"] == "r1"