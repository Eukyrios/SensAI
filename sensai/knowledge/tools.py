import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from . import japanese
DATA_DIR = Path(__file__).resolve().parent / "data"
VERB_CLASSES = {"ichidan", "godan", "suru", "kuru"}


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

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def subset(self, names: Iterable[str]) -> "ToolRegistry":
        chosen = ToolRegistry()
        for name in names:
            tool = self.get(name)
            if tool is not None:
                chosen.register(tool)
        return chosen

    def execute(self, name: str, args: dict) -> str:
        return self.run(name, args)[0]

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a tool and say whether it worked; a failure is a message for the model, never an exception."""
        tool = self.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found.", False
        if not isinstance(args, dict):
            return f"Error executing {name}: arguments must be a JSON object", False
        try:
            result = tool.func(**args)
        except Exception as e:
            return f"Error executing {name}: {str(e)}", False
        ok = not (isinstance(result, dict) and "error" in result)
        return json.dumps(result, ensure_ascii=False, default=str), ok

    def get_native_tools(self) -> list[dict]:
        return [tool.to_native_schema() for tool in self._tools.values()]

    def get_prompt_instructions(self) -> str:
        instructions = "You have access to the following tools:\n"
        for tool in self._tools.values():
            instructions += f"- {tool.name}: {tool.description}\n"
            instructions += f"  Arguments: {json.dumps(tool.parameters)}\n"
        instructions += '\nTo use a tool, output exactly a JSON block like this: {"tool": "name", "args": {}}'
        return instructions


# --- Own tools, backed by the local data files ---

@lru_cache(maxsize=1)
def _dictionary() -> tuple[dict, ...]:
    entries = json.loads((DATA_DIR / "dictionary.json").read_text(encoding="utf-8"))
    return tuple({**entry, "romaji": japanese.particle_romaji(entry["reading"], entry["pos"])} for entry in entries)


@lru_cache(maxsize=1)
def _kanji() -> dict[str, dict]:
    table = json.loads((DATA_DIR / "kanji.json").read_text(encoding="utf-8"))
    return {row[0]: dict(zip(table["columns"], row)) for row in table["rows"]}


def find_entries(word: str) -> list[dict]:
    query = word.strip()
    lowered = query.lower().removeprefix("to ").strip()
    exact = [entry for entry in _dictionary()
             if query in (entry["word"], entry["reading"]) or lowered == entry["romaji"]]
    if exact or not lowered.isascii() or not lowered:
        return exact
    pattern = re.compile(rf"(^|[\s,(]){re.escape(lowered)}($|[\s,)])")
    matches = [entry for entry in _dictionary() if pattern.search(entry["meaning"].lower())]
    return sorted(matches, key=lambda entry: not _has_meaning(entry, lowered))


def _has_meaning(entry: dict, meaning: str) -> bool:
    return meaning in (part.strip().removeprefix("to ") for part in entry["meaning"].lower().split(","))


def lookup_dictionary(word: str) -> dict:
    entries = find_entries(word)
    if not entries and len(word.strip()) == 1 and word.strip() in _kanji():
        character = word.strip()
        words = [_entry_view(entry) for entry in _dictionary() if character in entry["word"]][:5]
        return {"word": word, "found": True, "kanji": _kanji()[character], "words_with_it": words}
    if not entries:
        return {"word": word, "found": False,
                "message": f"'{word}' is not in the local dictionary; say so instead of guessing."}
    shown = [_entry_view(entry) for entry in entries[:5]]
    return {"word": word, "found": True, **shown[0]} if len(shown) == 1 else \
        {"word": word, "found": True, "entries": shown}


def _entry_view(entry: dict) -> dict:
    view = {"word": entry["word"], "reading": f"{entry['reading']} ({entry['romaji']})",
            "meaning": entry["meaning"], "type": entry["pos"], "jlpt": entry["jlpt"]}
    if entry.get("note"):
        view["note"] = entry["note"]
    return view


def conjugate_verb(verb: str, form: str) -> dict:
    verbs = [entry for entry in find_entries(verb) if entry["pos"] in VERB_CLASSES]
    written = [entry for entry in verbs if entry["word"] == verb]
    verbs = written or verbs
    if len({entry["word"] for entry in verbs}) > 1:
        options = ", ".join(f"{entry['word']} ({entry['meaning']}, {entry['pos']})" for entry in verbs)
        return {"verb": verb, "error": f"'{verb}' is several verbs: {options}; pass the one meant, in kanji"}
    if verbs:
        entry = verbs[0]
        word, reading, verb_class, source = entry["word"], entry["reading"], entry["pos"], "dictionary"
    elif _is_kana(verb) and verb.endswith(tuple(japanese._ROWS)):
        word, reading = verb, verb
        verb_class, source = japanese.guess_class(verb, verb), "guessed from the ending"
    else:
        return {"verb": verb, "error": f"'{verb}' is not a verb this tool knows; write it in kana to conjugate it"}
    try:
        result = japanese.conjugate(word, reading, verb_class, form).as_dict()
    except ValueError as error:
        return {"verb": verb, "error": str(error)}
    return {**result, "class_source": source}


def _is_kana(text: str) -> bool:
    return bool(text) and all("぀" <= char <= "ヿ" for char in text)


def lookup_kanji(character: str) -> dict:
    found = [_kanji()[char] for char in character if char in _kanji()]
    unknown = [char for char in character if "一" <= char <= "鿿" and char not in _kanji()]
    if not found:
        return {"kanji": character, "found": False,
                "message": f"'{character}' is not in the local kanji table; say so instead of guessing."}
    if len(found) == 1 and not unknown:
        return {**found[0], "found": True}
    return {"kanji": character, "found": True, "characters": found, "unknown": unknown}


def own_tools() -> list[Tool]:
    return [
        Tool(
            name="dictionary",
            description=("Look up a Japanese word (kanji, kana or romaji) or an English meaning in the local "
                         "dictionary: reading, meaning, word type, JLPT level and particle notes."),
            parameters={"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
            func=lookup_dictionary,
        ),
        Tool(
            name="conjugate",
            description=("Conjugate a Japanese verb. Forms: " + ", ".join(japanese.FORMS) + "."),
            parameters={"type": "object",
                        "properties": {"verb": {"type": "string", "description": "dictionary form, e.g. 食べる"},
                                       "form": {"type": "string", "description": "e.g. te, polite, nai"}},
                        "required": ["verb", "form"]},
            func=conjugate_verb,
        ),
        Tool(
            name="kanji",
            description="Get the readings (on'yomi, kun'yomi), meaning, stroke count and JLPT level of kanji.",
            parameters={"type": "object", "properties": {"character": {"type": "string"}},
                        "required": ["character"]},
            func=lookup_kanji,
        ),
    ]


registry = ToolRegistry()
for _tool in own_tools():
    registry.register(_tool)


def get_tools_for_model(client, model: str, tool_registry: ToolRegistry | None = None) -> dict:
    chosen = registry if tool_registry is None else tool_registry
    capabilities = client.capabilities(model)
    if "tools" in capabilities:
        return {"type": "native", "tools": chosen.get_native_tools()}
    return {"type": "prompted", "instructions": chosen.get_prompt_instructions()}
