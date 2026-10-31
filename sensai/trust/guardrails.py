"""Input and Output Guardrails (EV2).
Filters input prompts for prompt injection attacks and PII leakage.
Redacts or blocks outputs containing sensitive personal information.
Rules are read from the [guardrails] section of config.toml.
"""
from dataclasses import dataclass
import re
from typing import Callable, Literal
from ..core.events import Event, GuardrailTriggered
ActionType = Literal["allow", "redact", "block"]

BLOCKED_MARK = "[BLOCKED]"


@dataclass
class GuardrailResult:
    action: ActionType
    filtered_text: str
    rule_triggered: str | None = None
    reason: str | None = None


# --- PII Patterns ---
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
API_KEY_PATTERN = re.compile(r"\b(?:sk-[a-zA-Z0-9_-]{16,}|ghp_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16})\b")
PHONE_PATTERNS = (
    re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?!\d)"),
    re.compile(r"(?<![\w+])(?:00\d{2}[ .-]?|0)[1-9](?:[ .-]?\d{2}){4}(?!\d)"),
    re.compile(r"(?<!\d)0[789]0[ -]?\d{4}[ -]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)0\d{1,3}-\d{2,4}-\d{4}(?!\d)"),
)
PHONE_PATTERN = PHONE_PATTERNS[0]
MIN_PHONE_DIGITS = 9
MAX_PHONE_DIGITS = 15

# --- Injection Patterns ---
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are\s+now", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+(?:DAN|developer)\s+mode", re.IGNORECASE),
    re.compile(r"override\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    # A fake chat/system tag, as in "</user><system>New instructions: ...".
    re.compile(r"<\s*/?\s*(?:system|assistant|user)\s*>", re.IGNORECASE),
    re.compile(r"ignorez?\s+(?:toutes?\s+)?(?:les\s+)?instructions\s+pr[ée]c[ée]dentes", re.IGNORECASE),
    re.compile(r"(?:以前|前|上記)の(?:指示|命令|ルール)を(?:すべて)?(?:無視|忘れ)"),
    re.compile(r"(?:指示|命令|ルール)を(?:すべて|全て|全部)?(?:無視|忘れて)"),
    re.compile(r"forget\s+(?:all\s+)?(?:your|the|previous|prior)\s+(?:rules|instructions|prompts?)", re.IGNORECASE),
]

# --- Scope (the tutor stays a Japanese tutor) ---
LEAVES_THE_SUBJECT = re.compile(
    r"\b(?:forget|stop|drop|skip)\b[^.?!\n]{0,25}\b(?:japanese|the\s+lesson|teaching|being\s+a\s+(?:teacher|tutor))\b", re.IGNORECASE)
ABOUT_JAPANESE = re.compile(
    r"japan|nihongo|kanji|hiragana|katakana|romaji|furigana|kana\b|jlpt|keigo|particle|[\u3040-\u30ff\u4e00-\u9fff]",
    re.IGNORECASE)
OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\b(?:write|create|generate|implement|code|debug|fix)\b[^.?!\n]{0,60}"
               r"\b(?:python|javascript|typescript|java|c\+\+|c#|rust|golang|sql|html|css|bash|regex|algorithm|"
               r"source\s+code|script|program|code|function\s+that)\b", re.IGNORECASE),
    re.compile(r"\bsolve\b[^.?!\n]{0,40}\b(?:equation|integral|derivative|inequality|system\s+of)\b", re.IGNORECASE),
    re.compile(r"(?<![\w.])\d*[a-z]\s*[+\-]\s*\d+\s*=\s*\d+"),
    re.compile(r"\b(?:chemistry|physics|biology|geography|history|economics)\s+(?:homework|exam|exercise)", re.IGNORECASE),
]

REFUSALS = {
    "prompt_injection": "I can't follow that: it tries to change my instructions. "
                        "Let's get back to your Japanese lesson.",
    "out_of_scope": "I'm your Japanese tutor, so I can't help with that. "
                    "Ask me about vocabulary, grammar, kanji or practice sentences.",
}

DEFAULT_RULES = {
    "enabled": True,
    "block_prompt_injection": True,
    "block_out_of_scope": True,
    "redact_pii_input": True,
    "redact_pii_output": True,
}


def rules_from(config: dict | None) -> dict:
    """The guardrail rules: the [guardrails] section of a full config, or a flat dict of rules."""
    config = config or {}
    section = config.get("guardrails", config)
    if not isinstance(section, dict):
        section = {}
    return {key: section.get(key, default) for key, default in DEFAULT_RULES.items()}


def _mask_phones(text: str) -> str:
    def mask(match: re.Match) -> str:
        digits = sum(char.isdigit() for char in match.group())
        return "[REDACTED_PHONE]" if MIN_PHONE_DIGITS <= digits <= MAX_PHONE_DIGITS else match.group()
    for pattern in PHONE_PATTERNS:
        text = pattern.sub(mask, text)
    return text


def out_of_scope(text: str) -> bool:
    if LEAVES_THE_SUBJECT.search(text):
        return True
    if ABOUT_JAPANESE.search(text):
        return False
    return any(pattern.search(text) for pattern in OUT_OF_SCOPE_PATTERNS)


def refusal_for(rule: str | None) -> str:
    return REFUSALS.get(rule or "", REFUSALS["out_of_scope"])


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", text)
    text = API_KEY_PATTERN.sub("[REDACTED_KEY]", text)
    return _mask_phones(text)


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
        rules = rules_from(config)
        if not rules["enabled"]:
            return GuardrailResult(action="allow", filtered_text=text)

        # 1. Check for Prompt Injection
        if rules["block_prompt_injection"]:
            for pattern in PROMPT_INJECTION_PATTERNS:
                if pattern.search(text):
                    self._emit_trigger(turn_id, "input", "prompt_injection", "block")
                    return GuardrailResult(
                        action="block",
                        filtered_text=f"{BLOCKED_MARK} Input contains prompt injection attempt.",
                        rule_triggered="prompt_injection",
                        reason="Prompt injection pattern detected.",
                    )

        # 2. Check that the request is about learning Japanese
        if rules["block_out_of_scope"] and out_of_scope(text):
            self._emit_trigger(turn_id, "input", "out_of_scope", "block")
            return GuardrailResult(
                action="block",
                filtered_text=f"{BLOCKED_MARK} Input is out of scope for a Japanese tutor.",
                rule_triggered="out_of_scope",
                reason="Not about learning Japanese.",
            )

        # 3. Check for PII in input
        if rules["redact_pii_input"]:
            filtered_text = redact_pii(text)
            if filtered_text != text:
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
        rules = rules_from(config)
        if not rules["enabled"] or not rules["redact_pii_output"]:
            return GuardrailResult(action="allow", filtered_text=text)

        filtered_text = redact_pii(text)
        if filtered_text != text:
            self._emit_trigger(turn_id, "output", "pii_leakage", "redact")
            return GuardrailResult(
                action="redact",
                filtered_text=filtered_text,
                rule_triggered="pii_leakage",
                reason="PII redacted from output.",
            )

        return GuardrailResult(action="allow", filtered_text=text)


# --- Interface used by the turn loop and the functional wrappers ---

_default_engine = GuardrailEngine()


def check_input(text: str, config: dict | None = None) -> GuardrailResult:
    return _default_engine.filter_input(text, config)

def check_output(text: str, config: dict | None = None) -> GuardrailResult:
    return _default_engine.filter_output(text, config)

def filter_input(text: str, config: dict | None = None) -> str:
    return _default_engine.filter_input(text, config).filtered_text

def filter_output(text: str, config: dict | None = None) -> str:
    return _default_engine.filter_output(text, config).filtered_text