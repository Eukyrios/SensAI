# Session: one learner conversation with its agent, stored history, profile and model client.

from dataclasses import replace

from ..core.agent_spec import AgentSpec, spec_for
from ..core.ids import new_id
from ..core.persona import Persona, persona_for
from ..core.messages import Message
from ..llm.ollama import OllamaClient
from ..memory.prompts import PromptStore
from ..memory.repository import SessionRepository, open_repository
from .router import Router, load_router


class Session:

    def __init__(self, config: dict, client: OllamaClient | None = None,
                 session_id: str | None = None, repo: SessionRepository | None = None):
        self.config = config
        self.id = session_id or new_id()
        self.spec: AgentSpec = spec_for(config["agents"]["default"], config)
        ollama_settings = config["ollama"]
        self.client = client or OllamaClient(ollama_settings["url"], ollama_settings["timeout_s"])
        self.router: Router | None = load_router(config, self.client)
        self.repo = repo or open_repository(config)
        self.prompts = PromptStore(self.repo.conn)
        self.persona: Persona = persona_for(config["personas"]["default"], config)
        self.profile: Message | None = self.repo.profile_system_message(config["memory"]["profile"])
        self.history: list[Message] = []
        self.leaf: str | None = None
        self.stored = False
        self.summary = ""
        self.summarized = 0

    @classmethod
    def resume(cls, config: dict, session_id: str, client: OllamaClient | None = None,
               repo: SessionRepository | None = None) -> "Session":
        session = cls(config, client, session_id, repo)
        session.leaf = session.repo.get_session(session_id)["active_leaf_id"]
        session.history = session.repo.load_history(session_id)
        session.stored = True
        return session

    def record(self, role: str, content: str) -> str:
        if not self.stored:
            self.repo.create_session(self.spec.model, session_id=self.id)
            self.stored = True
        self.leaf = self.repo.add_message(self.id, role, content, self.leaf)
        return self.leaf

    def discard(self, message_id: str, parent_id: str | None) -> None:
        self.repo.discard_leaf(message_id)
        self.leaf = parent_id

    def set_persona(self, name: str) -> None:
        self.persona = persona_for(name, self.config)

    def set_model(self, model: str) -> None:
        self.client.require_model(model)
        think = self.spec.think and self.can_think(model)
        self.spec = replace(self.spec, model=model, think=think)

    def set_thinking(self, enabled: bool) -> None:
        if enabled and not self.can_think(self.spec.model):
            raise ValueError(f"{self.spec.model} can't reason before answering")
        self.spec = replace(self.spec, think=enabled)

    def can_think(self, model: str) -> bool:
        return "thinking" in self.client.capabilities(model)

    def reset(self) -> None:
        self.history.clear()
        self.id = new_id()
        self.leaf = None
        self.stored = False
        self.summary = ""
        self.summarized = 0
