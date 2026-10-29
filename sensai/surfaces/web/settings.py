# Web routes that change a session's settings mid-conversation, as the CLI's /model and /persona do:
# model, persona, and the agent (a named agent turns routing off, like --agent; none turns it back on).
# The history is kept in every case.

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from ...core.agent_spec import SpecError, spec_for
from ...core.persona import persona_for, persona_names
from ...llm.ollama import OllamaError
from ...orchestration.router import load_router
from ...orchestration.session import Session
from .sessions import SessionPool, open_session, session_view


class ModelChoice(BaseModel):
    model: str


class AgentChoice(BaseModel):
    agent: str | None = None


class PersonaChoice(BaseModel):
    persona: str


def add_settings_routes(app: FastAPI) -> None:
    app.add_api_route("/api/models", list_models, methods=["GET"])
    app.add_api_route("/api/personas", list_personas, methods=["GET"])
    app.add_api_route("/api/sessions/{session_id}/model", switch_model, methods=["PUT"])
    app.add_api_route("/api/sessions/{session_id}/agent", switch_agent, methods=["PUT"])
    app.add_api_route("/api/sessions/{session_id}/persona", switch_persona, methods=["PUT"])


def list_models(request: Request) -> dict:
    try:
        return {"installed": request.app.state.sessions.client.list_models()}
    except OllamaError as error:
        raise HTTPException(503, str(error)) from None


def list_personas(request: Request) -> list[dict]:
    config = request.app.state.sessions.config
    return [{"name": name, "description": persona_for(name, config).description} for name in persona_names(config)]


def switch_model(session_id: str, choice: ModelChoice, request: Request) -> dict:
    session = open_session(request.app.state.sessions, session_id)
    try:
        session.set_model(choice.model)
    except OllamaError as error:
        raise HTTPException(400 if error.kind == "model_missing" else 503, str(error)) from None
    return session_view(session)


def switch_agent(session_id: str, choice: AgentChoice, request: Request) -> dict:
    pool: SessionPool = request.app.state.sessions
    session = open_session(pool, session_id)
    try:
        if choice.agent:
            pin_agent(session, choice.agent)
        else:
            route_again(session, pool)
    except SpecError as error:
        raise HTTPException(404, str(error)) from None
    return session_view(session)


def pin_agent(session: Session, agent: str) -> None:
    session.spec = spec_for(agent, session.config)
    session.router = None


def route_again(session: Session, pool: SessionPool) -> None:
    session.spec = spec_for(session.config["agents"]["default"], session.config)
    session.router = load_router(session.config, pool.client)


def switch_persona(session_id: str, choice: PersonaChoice, request: Request) -> dict:
    session = open_session(request.app.state.sessions, session_id)
    try:
        session.set_persona(choice.persona)
    except SpecError as error:
        raise HTTPException(404, str(error)) from None
    return session_view(session)
