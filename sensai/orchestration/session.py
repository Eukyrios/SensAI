"""One learner conversation: the active agent, the history (stored in SQLite) and the model client."""
from dataclasses import replace

from ..core.agent_spec import AgentSpec, spec_for
from ..core.ids import new_id
from ..core.messages import Message
from ..llm.ollama import OllamaClient
from ..memory.repository import SessionRepository, open_repository


class Session:
    """A new session is only written to the database with its first message, so launching and
    quitting leaves no empty sessions behind. Use `Session.resume` to continue a stored one.
    """

    def __init__(self, config: dict, client: OllamaClient | None = None,
                 session_id: str | None = None, repo: SessionRepository | None = None):
        self.config = config
        self.id = session_id or new_id()
        self.spec: AgentSpec = spec_for(config["agents"]["default"], config)
        o = config["ollama"]
        self.client = client or OllamaClient(o["url"], o["timeout_s"])
        self.repo = repo or open_repository(config)
        # Added to every prompt but never written to `messages`.
        self.profile: Message | None = self.repo.profile_system_message(config["memory"]["profile"])
        self.history: list[Message] = []
        self.leaf: str | None = None      # id of the last stored message of the active branch
        self.stored = False

    @classmethod
    def resume(cls, config: dict, session_id: str, client: OllamaClient | None = None,
               repo: SessionRepository | None = None) -> "Session":
        """Reload a stored session. Raises SessionNotFound if the id is unknown."""
        session = cls(config, client, session_id, repo)
        session.leaf = session.repo.get_session(session_id)["active_leaf_id"]
        session.history = session.repo.load_history(session_id)
        session.stored = True
        return session

    def record(self, role: str, content: str) -> str:
        """Store a message at the end of the active branch and return its id."""
        if not self.stored:
            self.repo.create_session(self.spec.model, session_id=self.id)
            self.stored = True
        self.leaf = self.repo.add_message(self.id, role, content, self.leaf)
        return self.leaf

    def discard(self, message_id: str, parent_id: str | None) -> None:
        """Remove a stored message that never got an answer; the branch ends at `parent_id` again."""
        self.repo.discard_leaf(message_id)
        self.leaf = parent_id

    def set_model(self, model: str) -> None:
        """Switch model mid-session, keeping the history. Raises OllamaError if it isn't installed.

        Thinking is turned off if it was on and the new model can't reason.
        """
        self.client.require_model(model)
        think = self.spec.think and self.can_think(model)
        self.spec = replace(self.spec, model=model, think=think)

    def set_thinking(self, enabled: bool) -> None:
        """Turn the model's reasoning on or off. Raises ValueError if the model can't reason."""
        if enabled and not self.can_think(self.spec.model):
            raise ValueError(f"{self.spec.model} can't reason before answering")
        self.spec = replace(self.spec, think=enabled)

    def can_think(self, model: str) -> bool:
        return "thinking" in self.client.capabilities(model)

    def reset(self) -> None:
        """Start a fresh conversation; the old one stays stored."""
        self.history.clear()
        self.id = new_id()
        self.leaf = None
        self.stored = False
