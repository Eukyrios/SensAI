from dataclasses import dataclass, replace
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .config import load_toml, resolve

REQUIRED = ("name", "model", "num_ctx")


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class AgentSpec:
    """One agent, loaded from agents/<name>.toml. model and num_ctx are always explicit."""
    name: str
    model: str
    num_ctx: int
    system_prompt: str = ""
    temperature: float = 0.3
    think: bool = False
    tools: tuple[str, ...] = ()
    output_schema: dict | None = None
    timeout_s: int = 60


def load_spec(path: str | Path) -> AgentSpec:
    data = load_toml(path)
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise SpecError(f"{path}: missing required field(s): {', '.join(missing)}")
    try:
        spec = AgentSpec(**{**data, "tools": tuple(data.get("tools", ()))})
    except TypeError as e:
        raise SpecError(f"{path}: {e}") from None
    if spec.output_schema is not None:
        try:
            Draft202012Validator.check_schema(spec.output_schema)
        except SchemaError as e:
            raise SpecError(f"{path}: invalid output_schema: {e.message}") from None
    return spec


def spec_for(name: str, config: dict) -> AgentSpec:
    """Load an agent by name from the configured directory, applying agents.model_override."""
    agents = config["agents"]
    path = resolve(agents["dir"]) / f"{name}.toml"
    if not path.exists():
        raise SpecError(f"unknown agent '{name}' (no {path})")
    spec = load_spec(path)
    return replace(spec, model=agents["model_override"]) if agents.get("model_override") else spec
