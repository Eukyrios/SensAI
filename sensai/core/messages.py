from typing import Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict):
    """One chat message, in the format the Ollama /api/chat endpoint expects."""
    role: Role
    content: str
