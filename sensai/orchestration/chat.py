"""Base single-agent loop (B1) with in-session history. Replaced by run_turn() in layer 2."""
from collections.abc import Iterator
from dataclasses import replace

from ..core.agent_spec import AgentSpec, spec_for
from ..core.messages import Message
from ..llm.ollama import OllamaClient
from ..trust import guardrails


class Chat:
    def __init__(self, config: dict, client: OllamaClient | None = None):
        self.config = config
        self.spec: AgentSpec = spec_for(config["agents"]["default"], config)
        o = config["ollama"]
        self.client = client or OllamaClient(o["url"], o["timeout_s"])
        self.history: list[Message] = []

    def send(self, message: str) -> Iterator[str]:
        """Send one user message and stream the reply token by token."""
        message = message.strip()
        if not message:
            raise ValueError("Empty message")

        self.history.append({"role": "user", "content": guardrails.filter_input(message, self.config)})
        max_turns = self.config["memory"]["max_history_turns"]
        messages: list[Message] = (
            [{"role": "system", "content": self.spec.system_prompt}]
            + self.history[-max_turns * 2:]
        )

        spec = self.spec
        reply = ""
        try:
            for token in self.client.chat_stream(spec.model, messages, num_ctx=spec.num_ctx,
                                                 temperature=spec.temperature, think=spec.think):
                reply += token
                yield token
        except BaseException:
            self.history.pop()      # drop the unanswered message
            raise
        self.history.append({"role": "assistant", "content": reply})

    def set_model(self, model: str) -> None:
        """Switch model mid-session, keeping the history. Raises OllamaError if it isn't installed."""
        self.client.require_model(model)
        self.spec = replace(self.spec, model=model)

    def reset(self) -> None:
        self.history.clear()
