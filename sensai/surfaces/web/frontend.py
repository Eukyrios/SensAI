# Serves the single-page UI (frontend/index.html) from the same origin as the API.

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

INDEX = Path(__file__).resolve().parents[3] / "frontend" / "index.html"


def add_frontend_route(app: FastAPI) -> None:
    app.add_api_route("/", serve_index, methods=["GET"], response_class=HTMLResponse, include_in_schema=False)


def serve_index() -> HTMLResponse:
    if not INDEX.is_file():
        return HTMLResponse("<h1>Sensai</h1><p>frontend/index.html not found.</p>", status_code=404)
    return HTMLResponse(INDEX.read_text(encoding="utf-8"))
