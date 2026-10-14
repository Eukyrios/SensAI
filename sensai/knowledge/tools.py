# Tool registry and native/prompted tool calling, with the dictionary, conjugate and kanji tools.

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_native_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, args: dict) -> str:
        tool = self.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            result = tool.func(**args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    def get_native_tools(self) -> list[dict]:
        return [tool.to_native_schema() for tool in self._tools.values()]

    def get_prompt_instructions(self) -> str:
        instructions = "You have access to the following tools:\n"
        for tool in self._tools.values():
            instructions += f"- {tool.name}: {tool.description}\n"
            instructions += f"  Arguments: {json.dumps(tool.parameters)}\n"
        instructions += "\nTo use a tool, output exactly a JSON block like this: {\"tool\": \"name\", \"args\": {}}"
        return instructions


def lookup_dictionary(word: str) -> dict:
    return {"word": word, "meaning": "Définition factice pour le moment"}

def conjugate_verb(verb: str, form: str) -> dict:
    return {"verb": verb, "form": form, "result": f"{verb} conjugué au {form}"}

def lookup_kanji(character: str) -> dict:
    return {"kanji": character, "onyomi": "ON", "kunyomi": "kun", "meaning": "sens"}


registry = ToolRegistry()

registry.register(Tool(
    name="dictionary",
    description="Look up a Japanese word to get its meaning and reading.",
    parameters={"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
    func=lookup_dictionary
))

registry.register(Tool(
    name="conjugate",
    description="Conjugate a Japanese verb into a specific form.",
    parameters={"type": "object", "properties": {"verb": {"type": "string"}, "form": {"type": "string"}}, "required": ["verb", "form"]},
    func=conjugate_verb
))

registry.register(Tool(
    name="kanji",
    description="Get information about a specific Kanji character.",
    parameters={"type": "object", "properties": {"character": {"type": "string"}}, "required": ["character"]},
    func=lookup_kanji
))

def get_tools_for_model(client, model: str) -> dict:
    capabilities = client.capabilities(model)
    if "tools" in capabilities:
        return {"type": "native", "tools": registry.get_native_tools()}
    else:
        return {"type": "prompted", "instructions": registry.get_prompt_instructions()}
