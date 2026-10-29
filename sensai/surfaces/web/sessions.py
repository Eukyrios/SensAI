# The web app's open sessions: created on demand, resumed from storage the first time a stored
# session is asked for, and kept in memory with their model, persona and history.

from ...llm.ollama import OllamaClient
from ...memory.repository import open_repository
from ...orchestration.session import Session


class SessionPool:

    def __init__(self, config: dict, client: OllamaClient):
        self.config = config
        self.client = client
        self.repo = open_repository(config)
        self.open: dict[str, Session] = {}
        self.busy: set[str] = set()

    def new(self) -> Session:
        session = Session(self.config, self.client, repo=self.repo)
        self.open[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        if session_id not in self.open:
            self.open[session_id] = self._resume(session_id)
        return self.open[session_id]

    def _resume(self, session_id: str) -> Session:
        self.repo.get_session(session_id)
        return Session.resume(self.config, session_id, self.client, self.repo)
