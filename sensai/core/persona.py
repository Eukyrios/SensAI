# Persona (A5): the voice of the agents that talk to the learner, loaded from personas/<name>.toml
# (name, description, tone, scope). Applying one appends it to an agent's system prompt, so each
# persona gets its own prompt hash; agents with structured output are left as they are.

import re
from dataclasses import dataclass, replace
from pathlib import Path

from .agent_spec import AgentSpec, SpecError
from .config import load_toml, resolve

REQUIRED = ("name", "description", "tone", "scope")
PERSONA_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    tone: str
    scope: str

    def apply(self, spec: AgentSpec) -> AgentSpec:
        if spec.output_schema is not None:
            return spec
        return replace(spec, system_prompt=f"{spec.system_prompt}\n\n{self.instructions()}")

    def instructions(self) -> str:
        return (f"Persona: {self.name}.\nTone: {self.tone}\nScope: {self.scope}\n"
                "The persona changes your tone only: keep every rule above, including the format of the Japanese.")


def load_persona(path: str | Path) -> Persona:
    data = load_toml(path)
    missing = [field for field in REQUIRED if field not in data]
    if missing:
        raise SpecError(f"{path}: missing required field(s): {', '.join(missing)}")
    try:
        return Persona(**data)
    except TypeError as error:
        raise SpecError(f"{path}: {error}") from None


def persona_for(name: str, config: dict) -> Persona:
    if not PERSONA_NAME.fullmatch(name):
        raise SpecError(f"invalid persona name '{name}'")
    path = resolve(config["personas"]["dir"]) / f"{name}.toml"
    if not path.exists():
        raise SpecError(f"unknown persona '{name}' (no {path})")
    return load_persona(path)


def persona_names(config: dict) -> list[str]:
    return sorted(path.stem for path in resolve(config["personas"]["dir"]).glob("*.toml"))
