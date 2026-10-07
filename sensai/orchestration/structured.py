"""Structured output (T5): an agent with an output_schema must reply with JSON matching it.

Ollama's own schema enforcement can't be relied on. With qwen3.5 and think=false (our default),
Ollama 0.18 ignores a schema passed in `format` and returns free prose; it only enforces it when
the model reasons first, which takes about a minute per reply. So:

- the schema is written into the system prompt, which the model follows well;
- `format` is "json" when thinking is off (the reply is always parseable JSON) and the schema
  itself when thinking is on (Ollama enforces it then);
- the reply is validated against the schema; on failure the errors are sent back once, and a
  second failure is reported as an Error event with the invalid output.
"""
import json

from jsonschema import Draft202012Validator

from ..core.agent_spec import AgentSpec

MAX_SHOWN = 500         # characters of an invalid reply quoted in the error


def instructions(schema: dict) -> str:
    """Appended to the system prompt of an agent that has an output_schema."""
    return ("Reply with only one JSON object that matches this JSON schema. "
            "No other text, no explanations outside the JSON, no code fences.\n"
            + json.dumps(schema, ensure_ascii=False))


def request_format(spec: AgentSpec) -> dict | str | None:
    """What to send as Ollama's `format` for this agent (None: free text)."""
    if spec.output_schema is None:
        return None
    return spec.output_schema if spec.think else "json"


def check(text: str, schema: dict) -> tuple[dict | None, list[str]]:
    """Parse and validate a reply. Returns (data, []) when valid, (None, problems) otherwise."""
    try:
        data = json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON: {e.msg} (line {e.lineno}, column {e.colno})"]
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.absolute_path))
    problems = [f"{'/'.join(map(str, e.absolute_path)) or 'reply'}: {e.message}" for e in errors]
    return (None, problems) if problems else (data, [])


def repair_request(problems: list[str]) -> str:
    """The message sent back to the model for its one repair attempt."""
    return ("Your reply does not match the JSON schema:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\nReply again with only the corrected JSON object.")


def failure_message(problems: list[str], text: str) -> str:
    """The Error message when the repair attempt failed too: what was wrong, and the reply itself."""
    shown = text if len(text) <= MAX_SHOWN else text[:MAX_SHOWN] + "…"
    return ("The reply still doesn't match the schema after one repair attempt: "
            + "; ".join(problems) + f". Reply was: {shown}")


def _strip_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```").strip()
    return text
