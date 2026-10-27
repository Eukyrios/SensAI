# AgentSpec: one agent loaded from agents/<name>.toml, with its model, num_ctx and output schema.
# An agent that lists `handoffs` gets a `handoff` field in its output schema to pass the turn on.
# Its system prompt is versioned by content hash; `prompt_version` pins it to an earlier hash.

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .config import load_toml, resolve
from .handoff import handoff_schema

REQUIRED = ("name", "model", "num_ctx")
PROMPT_HASH_LENGTH = 12


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: str
    num_ctx: int
    system_prompt: str = ""
    temperature: float = 0.3
    think: bool = False
    tools: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()
    output_schema: dict | None = None
    timeout_s: int = 60
    prompt_version: str = ""

    @property
    def prompt_hash(self) -> str:
        return prompt_hash(self.system_prompt)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:PROMPT_HASH_LENGTH]


def load_spec(path: str | Path) -> AgentSpec:
    data = load_toml(path)
    missing = [field for field in REQUIRED if field not in data]
    if missing:
        raise SpecError(f"{path}: missing required field(s): {', '.join(missing)}")
    try:
        spec = AgentSpec(**{**data, "tools": tuple(data.get("tools", ())),
                            "handoffs": tuple(data.get("handoffs", ()))})
    except TypeError as error:
        raise SpecError(f"{path}: {error}") from None
    if spec.output_schema is not None:
        _check_schema(spec.output_schema, path)
    return _with_handoff_field(spec, path)


def _check_schema(schema: dict, path: str | Path) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise SpecError(f"{path}: invalid output_schema: {error.message}") from None


def _with_handoff_field(spec: AgentSpec, path: str | Path) -> AgentSpec:
    if not spec.handoffs:
        return spec
    if spec.output_schema is None or spec.output_schema.get("type") != "object":
        raise SpecError(f"{path}: an agent with handoffs needs an object output_schema")
    schema = dict(spec.output_schema)
    schema["properties"] = {**schema.get("properties", {}), "handoff": handoff_schema(spec.handoffs)}
    schema["required"] = [*schema.get("required", []), "handoff"]
    return replace(spec, output_schema=schema)


def spec_for(name: str, config: dict) -> AgentSpec:
    agents = config["agents"]
    path = resolve(agents["dir"]) / f"{name}.toml"
    if not path.exists():
        raise SpecError(f"unknown agent '{name}' (no {path})")
    spec = load_spec(path)
    return replace(spec, model=agents["model_override"]) if agents.get("model_override") else spec
