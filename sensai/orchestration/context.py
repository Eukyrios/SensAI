# Per-agent context and token budget (M2). Each provider renders one block of the system prompt
# (learner profile, handoff note, later the ledger and RAG); the builder gives each a share of the
# agent's num_ctx by priority, then fills what is left with the newest history. Token counts are
# estimated from the text and can be corrected from the prompt_eval_count Ollama reports.

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core.agent_spec import AgentSpec
from ..core.handoff import Handoff
from ..core.messages import Message
from .executor import system_prompt
from .session import Session

DEFAULT_CHARS_PER_TOKEN = 3.5
REPLY_RESERVE = 0.25


@runtime_checkable
class ContextProvider(Protocol):
    name: str
    priority: int
    max_share: float

    def render(self, budget: int) -> str | None: ...


@dataclass(frozen=True)
class TextBlock:
    name: str
    priority: int
    max_share: float
    text: str

    def render(self, budget: int) -> str | None:
        return self.text


class TokenEstimator:

    def __init__(self, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN, smoothing: float = 0.3):
        if chars_per_token <= 0 or not 0 < smoothing <= 1:
            raise ValueError("chars_per_token must be > 0 and smoothing in (0, 1]")
        self.chars_per_token = chars_per_token
        self.smoothing = smoothing

    def estimate(self, text: str) -> int:
        return max(1, round(len(text) / self.chars_per_token)) if text else 0

    def estimate_messages(self, messages: list[Message]) -> int:
        return sum(self.estimate(message["content"]) for message in messages)

    def calibrate(self, messages: list[Message], prompt_eval_count: int) -> None:
        chars = sum(len(message["content"]) for message in messages)
        if chars == 0 or prompt_eval_count <= 0:
            return
        observed = chars / prompt_eval_count
        self.chars_per_token += self.smoothing * (observed - self.chars_per_token)


@dataclass(frozen=True)
class Block:
    provider: str
    tokens: int
    truncated: bool


class ContextBuilder:

    def __init__(self, providers: list[ContextProvider] | None = None,
                 estimator: TokenEstimator | None = None):
        self.providers = list(providers or [])
        self.estimator = estimator or TokenEstimator()
        self.blocks: list[Block] = []

    def budget(self, spec: AgentSpec) -> int:
        return int(spec.num_ctx * (1 - REPLY_RESERVE))

    def build(self, spec: AgentSpec, history: list[Message], system: str | None = None) -> list[Message]:
        system = spec.system_prompt if system is None else system
        remaining = self.budget(spec) - self.estimator.estimate(system)
        self.blocks = []
        sections, remaining = self._render_blocks(max(remaining, 0), remaining)
        system_message = {"role": "system", "content": "\n\n".join(filter(None, [system, *sections]))}
        return [system_message] + self._latest(history, remaining)

    def calibrate(self, messages: list[Message], prompt_eval_count: int) -> None:
        self.estimator.calibrate(messages, prompt_eval_count)

    def _render_blocks(self, total: int, remaining: int) -> tuple[list[str], int]:
        sections = []
        for provider in sorted(self.providers, key=_highest_priority_first):
            cap = min(int(total * provider.max_share), remaining)
            text = provider.render(cap) if cap > 0 else None
            if not text:
                continue
            text, tokens, cut = self._fit(text, cap)
            sections.append(text)
            remaining -= tokens
            self.blocks.append(Block(provider.name, tokens, cut))
        return sections, remaining

    def _fit(self, text: str, cap: int) -> tuple[str, int, bool]:
        tokens = self.estimator.estimate(text)
        if tokens <= cap:
            return text, tokens, False
        text = text[:int(cap * self.estimator.chars_per_token)]
        return text, self.estimator.estimate(text), True

    def _latest(self, history: list[Message], budget: int) -> list[Message]:
        kept: list[Message] = []
        for message in reversed(history):
            cost = self.estimator.estimate(message["content"])
            if kept and cost > budget:
                break
            kept.append(message)
            budget -= cost
        return kept[::-1]


def build_context(session: Session, spec: AgentSpec, handoff: Handoff | None = None) -> list[Message]:
    max_turns = session.config["memory"]["max_history_turns"]
    builder = ContextBuilder(_blocks(session, handoff))
    return builder.build(spec, session.history[-max_turns * 2:], system=system_prompt(spec))


def handoff_note(handoff: Handoff) -> str:
    findings = json.dumps(handoff.payload, ensure_ascii=False)
    return (f"The {handoff.from_agent} agent looked at this turn first ({handoff.reason}). "
            f"Use its findings to answer the learner's last message: {findings}")


def _blocks(session: Session, handoff: Handoff | None) -> list[ContextProvider]:
    blocks = []
    if session.profile:
        blocks.append(TextBlock("profile", priority=2, max_share=0.2, text=session.profile["content"]))
    if handoff is not None:
        blocks.append(TextBlock("handoff", priority=1, max_share=0.3, text=handoff_note(handoff)))
    return blocks


def _highest_priority_first(provider: ContextProvider) -> int:
    return -provider.priority
