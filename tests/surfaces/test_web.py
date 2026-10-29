# Web UI backend: the FastAPI routes and the SSE encoding of the event stream.

import json

from fastapi.testclient import TestClient

from sensai.core.events import Token, TurnStarted
from sensai.surfaces.web import __main__ as web_main
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


def sse_names(response):
    return [line.removeprefix("event: ") for line in response.text.split("\n") if line.startswith("event: ")]


def sse_data(response, name):
    lines = response.text.split("\n")
    return [json.loads(lines[index + 1].removeprefix("data: "))
            for index, line in enumerate(lines) if line == f"event: {name}"]


def test_new_session_returns_its_id():
    response = make_client().post("/api/sessions")
    assert response.status_code == 201
    assert response.json()["id"]


def test_turn_streams_the_events_as_sse():
    client = make_client()
    session_id = client.post("/api/sessions").json()["id"]
    response = client.post(f"/api/sessions/{session_id}/turns", json={"text": "hello"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert sse_names(response)[0] == "TurnStarted"
    assert sse_names(response)[-1] == "TurnFinished"
    assert "".join(token["text"] for token in sse_data(response, "Token")) == "Konnichiwa"


def test_turn_in_an_unknown_session_is_404():
    response = make_client().post("/api/sessions/nope/turns", json={"text": "hello"})
    assert response.status_code == 404


def test_turn_while_another_runs_is_409():
    client = make_client()
    session_id = client.post("/api/sessions").json()["id"]
    client.app.state.sessions.busy.add(session_id)
    assert client.post(f"/api/sessions/{session_id}/turns", json={"text": "hello"}).status_code == 409


def test_a_finished_turn_frees_the_session():
    client = make_client()
    session_id = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{session_id}/turns", json={"text": "hello"})
    assert session_id not in client.app.state.sessions.busy


class ServeRecorder:

    def __init__(self):
        self.calls = []

    def __call__(self, app, **options):
        self.calls.append((app, options))


def test_main_serves_the_app_on_the_given_port(monkeypatch):
    serve = ServeRecorder()
    monkeypatch.setattr(web_main.uvicorn, "run", serve)
    web_main.main(["--port", "9000"])
    app, options = serve.calls[0]
    assert app.title == "Sensai"
    assert options == {"host": "127.0.0.1", "port": 9000}
