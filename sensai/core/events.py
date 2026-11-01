# The event stream: the only thing surfaces (CLI, logger, web UI) subscribe to.

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
    prompt_hash: str | None = None


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
class Thinking(_Base):
    run_id: str
    text: str


@dataclass(frozen=True, kw_only=True)
class StructuredOutput(_Base):
    run_id: str
    data: dict
    attempts: int = 1


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
class Retrieved(_Base):
    run_id: str
    query: str
    level: str | None
    mode: str
    kept: list[dict]
    dropped: list[dict]


@dataclass(frozen=True, kw_only=True)
class RouteDecided(_Base):
    agent: str
    tier: str
    confidence: float
    reason: str


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
    model: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0
    ttft_ms: float | None = None


@dataclass(frozen=True, kw_only=True)
class ContextCompressed(_Base):
    """Old turns were folded into the running summary (M2): `messages` turns, `tokens_before` -> `tokens_after`."""
    messages: int
    tokens_before: int
    tokens_after: int


@dataclass(frozen=True, kw_only=True)
class FaithfulnessChecked(_Base):
    """The judge's verdict on a reply, checked against the passages and tool results the agent had (EV1).
    `flagged` lists the claims the sources do not support: {"claim", "status", "reasoning"}."""
    faithfulness: float | None
    relevance: float | None = None
    supported: int = 0
    contradicted: int = 0
    unverified: int = 0
    flagged: list[dict] = field(default_factory=list)
    judge_model: str = ""
    error: str | None = None


Event = (TurnStarted | AgentRunStarted | AgentRunFinished | Thinking | Token | StructuredOutput
         | ToolCall | ToolResult | Retrieved | RouteDecided | HandoffEvent | GuardrailTriggered | Error | TurnFinished
         | FaithfulnessChecked | ContextCompressed)
