"""Tool registry and native/prompted dual path for the SensAI knowledge base."""
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
            },
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
        instructions += '\nTo use a tool, output exactly a JSON block like this: {"tool": "name", "args": {}}'
        return instructions


# --- Base de données locale déterministe pour les outils ---

_DICTIONARY_DB: dict[str, dict[str, Any]] = {
    "食べる": {"word": "食べる", "reading": "たべる (taberu)", "meaning": "to eat", "type": "Ichidan verb (ru-verb)", "jlpt": "N5"},
    "行く": {"word": "行く", "reading": "いく (iku)", "meaning": "to go (takes particle に or へ for destination, never を)", "type": "Godan verb (irregular te-form)", "jlpt": "N5"},
    "見る": {"word": "見る", "reading": "みる (miru)", "meaning": "to see, to watch", "type": "Ichidan verb (ru-verb)", "jlpt": "N5"},
    "飲む": {"word": "飲む", "reading": "のむ (nomu)", "meaning": "to drink", "type": "Godan verb (mu-verb)", "jlpt": "N5"},
    "学校": {"word": "学校", "reading": "がっこう (gakkou)", "meaning": "school", "type": "Noun", "jlpt": "N5"},
}

_CONJUGATION_DB: dict[str, dict[str, str]] = {
    "食べる": {"te": "食べて", "ta": "食べた", "nai": "食べない", "polite": "食べます", "past_negative_polite": "食べませんでした"},
    "taberu": {"te": "tabete (食べて)", "ta": "tabeta (食べた)", "nai": "tabenai (食べない)", "polite": "tabemasu (食べます)", "past_negative_polite": "tabemasen deshita (食べませんでした)"},
    "行く": {"te": "行って", "ta": "行った", "nai": "行かない", "polite": "行きます", "past_negative_polite": "行きませんでした"},
    "iku": {"te": "itte (行って)", "ta": "itta (行った)", "nai": "ikanai (行かない)", "polite": "ikimasu (行きます)", "past_negative_polite": "ikimasen deshita (行きませんでした)"},
    "見る": {"te": "見て", "ta": "見た", "nai": "見ない", "polite": "見ます", "past_negative_polite": "見ませんでした"},
    "飲む": {"te": "飲んで", "ta": "飲んだ", "nai": "飲まない", "polite": "飲みます", "past_negative_polite": "飲みませんでした"},
}

_KANJI_DB: dict[str, dict[str, Any]] = {
    "鬱": {"kanji": "鬱", "onyomi": "ウツ (utsu)", "kunyomi": "ふさ.ぐ (fusa.gu)", "meaning": "gloom, depression, melancholy", "stroke_count": 29},
    "食": {"kanji": "食", "onyomi": "ショク (shoku)", "kunyomi": "た.べる (ta.beru)", "meaning": "eat, food", "stroke_count": 9},
    "行": {"kanji": "行", "onyomi": "コウ (kou), ギョウ (gyou)", "kunyomi": "い.く (i.ku), おこな.う (okona.u)", "meaning": "go, journey, carry out", "stroke_count": 6},
    "学": {"kanji": "学", "onyomi": "ガク (gaku)", "kunyomi": "まな.ぶ (mana.bu)", "meaning": "study, learning, science", "stroke_count": 8},
    "校": {"kanji": "校", "onyomi": "コウ (kou)", "kunyomi": "-", "meaning": "school, exam", "stroke_count": 10},
}


def lookup_dictionary(word: str) -> dict:
    if word in _DICTIONARY_DB:
        return _DICTIONARY_DB[word]
    return {"word": word, "reading": "unknown", "meaning": f"Entry for '{word}'", "jlpt": "unknown"}


def conjugate_verb(verb: str, form: str) -> dict:
    normalized_form = form.lower().replace("-form", "").replace("form", "").strip()
    if verb in _CONJUGATION_DB and normalized_form in _CONJUGATION_DB[verb]:
        return {"verb": verb, "form": normalized_form, "result": _CONJUGATION_DB[verb][normalized_form]}
    if verb.endswith("る"):
        stem = verb[:-1]
        fallback_map = {"te": f"{stem}て", "ta": f"{stem}た", "nai": f"{stem}ない", "polite": f"{stem}ます"}
        if normalized_form in fallback_map:
            return {"verb": verb, "form": normalized_form, "result": fallback_map[normalized_form]}
    return {"verb": verb, "form": normalized_form, "result": f"{verb} ({normalized_form})"}


def lookup_kanji(character: str) -> dict:
    if character in _KANJI_DB:
        return _KANJI_DB[character]
    return {"kanji": character, "onyomi": "ON", "kunyomi": "kun", "meaning": "meaning", "stroke_count": 1}


registry = ToolRegistry()

registry.register(Tool(
    name="dictionary",
    description="Look up a Japanese word to get its meaning and reading.",
    parameters={"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
    func=lookup_dictionary,
))

registry.register(Tool(
    name="conjugate",
    description="Conjugate a Japanese verb into a specific form.",
    parameters={"type": "object", "properties": {"verb": {"type": "string"}, "form": {"type": "string"}}, "required": ["verb", "form"]},
    func=conjugate_verb,
))

registry.register(Tool(
    name="kanji",
    description="Get information about a specific Kanji character including readings and stroke count.",
    parameters={"type": "object", "properties": {"character": {"type": "string"}}, "required": ["character"]},
    func=lookup_kanji,
))


def get_tools_for_model(client, model: str) -> dict:
    """Détermine la voie d'exécution en fonction des capacités du modèle."""
    capabilities = client.capabilities(model)
    if "tools" in capabilities:
        return {"type": "native", "tools": registry.get_native_tools()}
    return {"type": "prompted", "instructions": registry.get_prompt_instructions()}