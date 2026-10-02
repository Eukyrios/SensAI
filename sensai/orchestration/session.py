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
        self.history.clear()
