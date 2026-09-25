"""The event stream: the only thing surfaces (CLI, logger, web UI) subscribe to.

Events are keyword-only so new fields can be added without breaking callers.
"""
import time
from dataclasses import dataclass, field
from typing import Literal

from .errors import ErrorKind
from .handoff import Handoff


@dataclass(frozen=True, kw_only=True)
class _Base:
    turn_id: str
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True, kw_only=True)
class TurnStarted(_Base):
    session_id: str
    user_input: str


@dataclass(frozen=True, kw_only=True)
class AgentRunStarted(_Base):
    run_id: str
    agent: str
    parent_run_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class AgentRunFinished(_Base):
    run_id: str
    agent: str
    ok: bool


@dataclass(frozen=True, kw_only=True)
class Token(_Base):
    run_id: str
    text: str


@dataclass(frozen=True, kw_only=True)
class ToolCall(_Base):
    run_id: str
    name: str
    args: dict


@dataclass(frozen=True, kw_only=True)
class ToolResult(_Base):
    run_id: str
    name: str
    result: str
    ok: bool = True


@dataclass(frozen=True, kw_only=True)
class HandoffEvent(_Base):
    handoff: Handoff


@dataclass(frozen=True, kw_only=True)
class GuardrailTriggered(_Base):
    side: Literal["input", "output"]
    rule: str
    action: Literal["block", "redact", "refuse"]


@dataclass(frozen=True, kw_only=True)
class Error(_Base):
    kind: ErrorKind
    message: str
    recoverable: bool = True


@dataclass(frozen=True, kw_only=True)
class TurnFinished(_Base):
    """Telemetry from Ollama's final chunk, using the API's field names."""
    model: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ns: int = 0
    ttft_ms: float | None = None


Event = (TurnStarted | AgentRunStarted | AgentRunFinished | Token | ToolCall | ToolResult
         | HandoffEvent | GuardrailTriggered | Error | TurnFinished)
