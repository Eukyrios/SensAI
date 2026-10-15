"""Per-agent context builder and token budget (M2).

A provider renders one block of the prompt (profile, ledger, RAG...). The builder gives each one a
share of the agent's context window, then fills what is left with the latest history turns. Token
counts are estimated from the text, then corrected from the real prompt_eval_count Ollama reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core.agent_spec import AgentSpec
from ..core.messages import Message

DEFAULT_CHARS_PER_TOKEN = 3.5       # between English (~4) and Japanese (~1.5), before calibration
REPLY_RESERVE = 0.25                # share of num_ctx kept free for the model's answer


@runtime_checkable
class ContextProvider(Protocol):
    """One block of the prompt.

    priority   higher is served first, and kept when the budget runs out
    max_share  largest fraction (0-1] of the prompt budget this block may take
    render     the block's text within `budget` tokens (the builder estimates its size), or None
    """
    name: str
    priority: int
    max_share: float

    def render(self, budget: int) -> str | None: ...


class TokenEstimator:
    """chars -> tokens, with a ratio corrected from real prompt_eval_count values."""

    def __init__(self, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN, smoothing: float = 0.3):
        if chars_per_token <= 0 or not 0 < smoothing <= 1:
            raise ValueError("chars_per_token must be > 0 and smoothing in (0, 1]")
        self.chars_per_token = chars_per_token
        self.smoothing = smoothing

    def estimate(self, text: str) -> int:
        return max(1, round(len(text) / self.chars_per_token)) if text else 0

    def estimate_messages(self, messages: list[Message]) -> int:
        return sum(self.estimate(m["content"]) for m in messages)

    def calibrate(self, messages: list[Message], prompt_eval_count: int) -> None:
        """Move the ratio toward what the model really counted. Ignores empty or missing counts."""
        chars = sum(len(m["content"]) for m in messages)
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
        self.blocks: list[Block] = []       # what the last build() included, for logs and tests

    def budget(self, spec: AgentSpec) -> int:
        """Tokens available for the prompt: the window minus the room kept for the answer."""
        return int(spec.num_ctx * (1 - REPLY_RESERVE))

    def build(self, spec: AgentSpec, history: list[Message], system: str | None = None) -> list[Message]:
        """System prompt, then each provider's block (by priority), then as much history as fits."""
        system = spec.system_prompt if system is None else system
        remaining = self.budget(spec) - self.estimator.estimate(system)
        total = max(remaining, 0)
        self.blocks = []

        sections: list[str] = []
        for p in sorted(self.providers, key=lambda p: -p.priority):
            cap = min(int(total * p.max_share), remaining)
            if cap <= 0:
                continue
            text = p.render(cap)
            if not text:
                continue
            text, tokens, cut = self._fit(text, cap)
            sections.append(text)
            remaining -= tokens
            self.blocks.append(Block(p.name, tokens, cut))

        messages: list[Message] = [{"role": "system", "content": "\n\n".join(filter(None, [system, *sections]))}]
        return messages + self._latest(history, remaining)

    def _fit(self, text: str, cap: int) -> tuple[str, int, bool]:
        tokens = self.estimator.estimate(text)
        if tokens <= cap:
            return text, tokens, False
        text = text[:int(cap * self.estimator.chars_per_token)]
        return text, self.estimator.estimate(text), True

    def _latest(self, history: list[Message], budget: int) -> list[Message]:
        """The newest messages that fit. The last one (the learner's question) is always kept."""
        kept: list[Message] = []
        for message in reversed(history):
            cost = self.estimator.estimate(message["content"])
            if kept and cost > budget:
                break
            kept.append(message)
            budget -= cost
        return kept[::-1]

    def calibrate(self, messages: list[Message], prompt_eval_count: int) -> None:
        self.estimator.calibrate(messages, prompt_eval_count)
