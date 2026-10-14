# Structured output: schema in the prompt, JSON validation and one repair attempt.

import json

from jsonschema import Draft202012Validator

from ..core.agent_spec import AgentSpec

MAX_SHOWN = 500


def schema_instructions(schema: dict) -> str:
    return ("Reply with only one JSON object that matches this JSON schema. "
            "No other text, no explanations outside the JSON, no code fences.\n"
            + json.dumps(schema, ensure_ascii=False))


def request_format(spec: AgentSpec) -> dict | str | None:
    if spec.output_schema is None:
        return None
    return spec.output_schema if spec.think else "json"


def validate_reply(text: str, schema: dict) -> tuple[dict | None, list[str]]:
    try:
        data = json.loads(_strip_fences(text))
    except json.JSONDecodeError as error:
        return None, [f"not valid JSON: {error.msg} (line {error.lineno}, column {error.colno})"]
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda error: list(error.absolute_path))
    problems = [_describe_validation_error(error) for error in errors]
    return (None, problems) if problems else (data, [])


def repair_request(problems: list[str]) -> str:
    return ("Your reply does not match the JSON schema:\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + "\nReply again with only the corrected JSON object.")


def failure_message(problems: list[str], text: str) -> str:
    shown = text if len(text) <= MAX_SHOWN else text[:MAX_SHOWN] + "…"
    return ("The reply still doesn't match the schema after one repair attempt: "
            + "; ".join(problems) + f". Reply was: {shown}")


def _describe_validation_error(error) -> str:
    field_path = "/".join(map(str, error.absolute_path)) or "reply"
    return f"{field_path}: {error.message}"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```").strip()
    return text
