"""Input and Output Guardrails (EV2).

Filters input prompts for prompt injection attacks and PII leakage.
Redacts or blocks outputs containing sensitive personal information.
"""
from dataclasses import dataclass
import re
from typing import Callable, Literal

from ..core.events import Event, GuardrailTriggered


ActionType = Literal["allow", "redact", "block"]


@dataclass
class GuardrailResult:
    action: ActionType
    filtered_text: str
    rule_triggered: str | None = None
    reason: str | None = None


# --- PII Patterns ---
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
API_KEY_PATTERN = re.compile(r"\b(?:sk-[a-zA-Z0-9]{16,}|ghp_[a-zA-Z0-9]{36}|[a-zA-Z0-9]{32})\b")

# --- Injection Patterns ---
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?prior\s+(?:prompts|rules)", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are\s+now", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+(?:DAN|developer)\s+mode", re.IGNORECASE),
    re.compile(r"override\s+system\s+prompt", re.IGNORECASE),
]


class GuardrailEngine:
    """Core evaluation engine for input and output guardrails."""

    def __init__(self, publisher: Callable[[Event], None] | None = None) -> None:
        self.publisher = publisher

    def _emit_trigger(self, turn_id: str, side: Literal["input", "output"], rule: str, action: Literal["block", "redact", "refuse"]) -> None:
        if self.publisher:
            self.publisher(
                GuardrailTriggered(
                    turn_id=turn_id,
                    side=side,
                    rule=rule,
                    action=action,
                )
            )

    def filter_input(self, text: str, config: dict | None = None, turn_id: str = "turn_unknown") -> GuardrailResult:
        """Evaluates user inputs against prompt injections and PII."""
        cfg = config or {}
        block_injection = cfg.get("block_prompt_injection", True)
        redact_pii = cfg.get("redact_pii_input", True)

        # 1. Check for Prompt Injection
        if block_injection:
            for pattern in PROMPT_INJECTION_PATTERNS:
                if pattern.search(text):
                    self._emit_trigger(turn_id, "input", "prompt_injection", "block")
                    return GuardrailResult(
                        action="block",
                        filtered_text="[BLOCKED] Input contains prompt injection attempt.",
                        rule_triggered="prompt_injection",
                        reason="Prompt injection pattern detected.",
                    )

        # 2. Check for PII in input
        filtered_text = text
        pii_found = False

        if redact_pii:
            if EMAIL_PATTERN.search(filtered_text):
                filtered_text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", filtered_text)
                pii_found = True
            if CREDIT_CARD_PATTERN.search(filtered_text):
                filtered_text = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", filtered_text)
                pii_found = True
            if API_KEY_PATTERN.search(filtered_text):
                filtered_text = API_KEY_PATTERN.sub("[REDACTED_KEY]", filtered_text)
                pii_found = True

            if pii_found:
                self._emit_trigger(turn_id, "input", "pii_leakage", "redact")
                return GuardrailResult(
                    action="redact",
                    filtered_text=filtered_text,
                    rule_triggered="pii_leakage",
                    reason="PII detected and redacted from input.",
                )

        return GuardrailResult(action="allow", filtered_text=text)

    def filter_output(self, text: str, config: dict | None = None, turn_id: str = "turn_unknown") -> GuardrailResult:
        """Evaluates model outputs to prevent secret or PII leakage."""
        cfg = config or {}
        redact_pii = cfg.get("redact_pii_output", True)

        if not redact_pii:
            return GuardrailResult(action="allow", filtered_text=text)

        filtered_text = text
        pii_found = False

        if EMAIL_PATTERN.search(filtered_text):
            filtered_text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", filtered_text)
            pii_found = True
        if CREDIT_CARD_PATTERN.search(filtered_text):
            filtered_text = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", filtered_text)
            pii_found = True
        if API_KEY_PATTERN.search(filtered_text):
            filtered_text = API_KEY_PATTERN.sub("[REDACTED_KEY]", filtered_text)
            pii_found = True

        if pii_found:
            self._emit_trigger(turn_id, "output", "pii_leakage", "redact")
            return GuardrailResult(
                action="redact",
                filtered_text=filtered_text,
                rule_triggered="pii_leakage",
                reason="PII redacted from output.",
            )

        return GuardrailResult(action="allow", filtered_text=text)


# --- Backward compatibility functional interface ---

_default_engine = GuardrailEngine()


def filter_input(text: str, config: dict | None = None) -> str:
    return _default_engine.filter_input(text, config).filtered_text

def filter_output(text: str, config: dict | None = None) -> str:
    return _default_engine.filter_output(text, config).filtered_text