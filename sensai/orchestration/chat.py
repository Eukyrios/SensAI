"""Agent central — propriétaire : A
Tout le monde (CLI, API) passe par cette classe."""
from typing import Iterator

from .interfaces import Message, PendingAction
from .ollama_client import OllamaClient
from . import guardrails

DEFAULT_SYSTEM = "Tu es Sensai, un assistant qui aide à gérer ses mails. Réponds en français."


class Agent:
    def __init__(self, config: dict, model: str | None = None):
        self.config = config
        o = config["ollama"]
        self.client = OllamaClient(
            url=o["url"],
            model=model or o["model"],
            think=o.get("think", False),
            temperature=o.get("temperature", 0.3),
            timeout=o.get("timeout", 60),
        )
        self.history: list[Message] = []
        self.persona = config.get("persona", {}).get("default", "formal")
        self._pending: PendingAction | None = None

    def _system_prompt(self) -> str:
        return DEFAULT_SYSTEM   # A5 : charger depuis personas/<self.persona>.yaml

    def chat(self, message: str) -> Iterator[str]:
        """Envoie un message et renvoie la réponse token par token."""
        message = message.strip()
        if not message:
            raise ValueError("Message vide")

        clean = guardrails.filter_input(message, self.config)
        self.history.append({"role": "user", "content": clean})

        max_turns = self.config.get("memory", {}).get("max_history_turns", 20)
        messages: list[Message] = (
            [{"role": "system", "content": self._system_prompt()}]
            + self.history[-max_turns * 2:]
        )

        reply = ""
        try:
            for token in self.client.chat_stream(messages):
                reply += token
                yield token
        except Exception:
            self.history.pop()      # on retire le message si échec
            raise

        self.history.append({"role": "assistant", "content": reply})

    def set_persona(self, name: str) -> None:
        """Change de persona sans effacer l'historique (A5)."""
        self.persona = name

    def pending_action(self) -> PendingAction | None:
        """Action en attente de confirmation (A3, C)."""
        return self._pending

    def confirm(self, approved: bool) -> str:
        """Valide ou annule l'action en attente (A3, C)."""
        self._pending = None
        return "Action confirmée." if approved else "Action annulée."

    def reset(self) -> None:
        self.history.clear()