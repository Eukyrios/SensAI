"""One learner conversation: the active agent, the in-session history and the model client."""
from dataclasses import replace

from ..core.agent_spec import AgentSpec, spec_for
from ..core.ids import new_id
from ..core.messages import Message
from ..llm.ollama import OllamaClient


class Session:
    def __init__(self, config: dict, client: OllamaClient | None = None,
                 session_id: str | None = None):
        self.config = config
        self.id = session_id or new_id()
        self.spec: AgentSpec = spec_for(config["agents"]["default"], config)
        o = config["ollama"]
        self.client = client or OllamaClient(o["url"], o["timeout_s"])
        self.history: list[Message] = []

    def set_model(self, model: str) -> None:
        """Switch model mid-session, keeping the history. Raises OllamaError if it isn't installed."""
        self.client.require_model(model)
        self.spec = replace(self.spec, model=model)

    def reset(self) -> None:
        self.history.clear()
