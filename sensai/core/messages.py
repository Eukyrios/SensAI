# Chat message format sent to the model.

from typing import Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict):
    role: Role
    content: str
