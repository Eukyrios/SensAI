# Web UI backend (X1): a FastAPI app over the event stream. It only subscribes to what the
# orchestration emits; the turn logic stays in sensai.orchestration.

from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...llm.ollama import OllamaClient
from ...memory.repository import SessionNotFound
from ...orchestration.session import Session
from ...orchestration.turn import run_turn
from .sessions import SessionPool
from .sse import sse_stream


class TurnRequest(BaseModel):
    text: str


def create_app(config: dict, client: OllamaClient | None = None) -> FastAPI:
    app = FastAPI(title="Sensai")
    client = client or OllamaClient(config["ollama"]["url"], config["ollama"]["timeout_s"])
    app.state.sessions = SessionPool(config, client)
    app.add_api_route("/api/health", health, methods=["GET"])
    app.add_api_route("/api/sessions", list_sessions, methods=["GET"])
    app.add_api_route("/api/sessions", new_session, methods=["POST"], status_code=201)
    app.add_api_route("/api/sessions/{session_id}", show_session, methods=["GET"])
    app.add_api_route("/api/sessions/{session_id}/turns", start_turn, methods=["POST"])
    return app


def health() -> dict:
    return {"status": "ok"}


def list_sessions(request: Request) -> list[dict]:
    rows = request.app.state.sessions.repo.list_sessions()
    return [{"id": row["id"], "created_at": row["created_at"], "model": row["model"], "title": row["title"]}
            for row in rows]


def new_session(request: Request) -> dict:
    return session_view(request.app.state.sessions.new())


def show_session(session_id: str, request: Request) -> dict:
    return session_view(open_session(request.app.state.sessions, session_id))


def session_view(session: Session) -> dict:
    return {"id": session.id, "agent": session.spec.name, "model": session.spec.model,
            "routing": session.router is not None, "persona": session.persona.name,
            "thinking": session.spec.think, "history": session.history}


def start_turn(session_id: str, turn: TurnRequest, request: Request) -> StreamingResponse:
    pool: SessionPool = request.app.state.sessions
    session = open_session(pool, session_id)
    if session_id in pool.busy:
        raise HTTPException(409, "a turn is already running in this session")
    pool.busy.add(session_id)
    return StreamingResponse(sse_stream(turn_events(pool, session, turn.text)), media_type="text/event-stream")


def open_session(pool: SessionPool, session_id: str) -> Session:
    try:
        return pool.get(session_id)
    except SessionNotFound:
        raise HTTPException(404, f"no session {session_id}") from None


async def turn_events(pool: SessionPool, session: Session, text: str) -> AsyncIterator:
    try:
        async for event in run_turn(session, text):
            yield event
    finally:
        pool.busy.discard(session.id)
