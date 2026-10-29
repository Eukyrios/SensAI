# Web UI backend (X1): a FastAPI app over the event stream. It only subscribes to what the
# orchestration emits; the turn logic stays in sensai.orchestration.

from fastapi import FastAPI

from ...llm.ollama import OllamaClient


def create_app(config: dict, client: OllamaClient | None = None) -> FastAPI:
    app = FastAPI(title="Sensai")
    app.state.config = config
    app.state.client = client or OllamaClient(config["ollama"]["url"], config["ollama"]["timeout_s"])
    app.add_api_route("/api/health", health, methods=["GET"])
    return app


def health() -> dict:
    return {"status": "ok"}
