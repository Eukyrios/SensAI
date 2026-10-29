# Web UI backend: the FastAPI routes and the SSE encoding of the event stream.

import json

from fastapi.testclient import TestClient

from sensai.core.events import Token, TurnStarted
from sensai.surfaces.web.app import create_app
from sensai.surfaces.web.sse import event_message, sse_stream
from tests.fakes import FakeClient, events_stream, make_config


def make_client(**config_sections):
    return TestClient(create_app(make_config(**config_sections), FakeClient()))


def test_health():
    assert make_client().get("/api/health").json() == {"status": "ok"}


def test_event_message_is_named_after_the_event():
    message = event_message(Token(turn_id="t1", run_id="r1", text="猫", ts=1.0))
    name_line, data_line, _, _ = message.split("\n")
    assert name_line == "event: Token"
    assert json.loads(data_line.removeprefix("data: ")) == {"turn_id": "t1", "ts": 1.0, "run_id": "r1", "text": "猫"}
    assert message.endswith("\n\n")


async def test_sse_stream_encodes_every_event():
    events = events_stream(TurnStarted(turn_id="t1", session_id="s1", user_input="hi"),
                           Token(turn_id="t1", run_id="r1", text="Kon"))
    messages = [message async for message in sse_stream(events)]
    assert [message.split("\n")[0] for message in messages] == ["event: TurnStarted", "event: Token"]
