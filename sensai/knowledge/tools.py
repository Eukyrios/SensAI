"""Tool registry and native/prompted dual path for the SensAI knowledge base."""
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema des arguments
    func: Callable[..., Any]

    def to_native_schema(self) -> dict:
        """Format required by Ollama's native tool calling."""
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
        """Exécute l'outil et renvoie le résultat en texte."""
        tool = self.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            result = tool.func(**args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    def get_native_tools(self) -> list[dict]:
        """Retourne la liste des outils pour l'API Ollama."""
        return [tool.to_native_schema() for tool in self._tools.values()]

    def get_prompt_instructions(self) -> str:
        """Retourne les instructions pour le fallback 'prompted' (modèles sans support natif)."""
        instructions = "You have access to the following tools:\n"
        for tool in self._tools.values():
            instructions += f"- {tool.name}: {tool.description}\n"
            instructions += f"  Arguments: {json.dumps(tool.parameters)}\n"
        instructions += "\nTo use a tool, output exactly a JSON block like this: {\"tool\": \"name\", \"args\": {}}"
        return instructions


# --- Implémentation des outils spécifiques ---

def lookup_dictionary(word: str) -> dict:
    # TODO: Connecter à un vrai dictionnaire ou un fichier local JSON
    return {"word": word, "meaning": "Définition factice pour le moment"}

def conjugate_verb(verb: str, form: str) -> dict:
    # TODO: Implémenter la logique de conjugaison japonaise
    return {"verb": verb, "form": form, "result": f"{verb} conjugué au {form}"}

def lookup_kanji(character: str) -> dict:
    # TODO: Récupérer les lectures On/Kun et la signification
    return {"kanji": character, "onyomi": "ON", "kunyomi": "kun", "meaning": "sens"}


# Initialisation du registre global
registry = ToolRegistry()

registry.register(Tool(
    name="dictionary",
    description="Look up a Japanese word to get its meaning and reading.",
    parameters={
        "type": "object",
        "properties": {
            "word": {"type": "string", "description": "The Japanese word to look up"}
        },
        "required": ["word"]
    },
    func=lookup_dictionary
))

registry.register(Tool(
    name="conjugate",
    description="Conjugate a Japanese verb into a specific form.",
    parameters={
        "type": "object",
        "properties": {
            "verb": {"type": "string", "description": "The dictionary form of the verb"},
            "form": {"type": "string", "description": "The target form (e.g., 'te', 'nai', 'ta', 'polite')"}
        },
        "required": ["verb", "form"]
    },
    func=conjugate_verb
))

registry.register(Tool(
    name="kanji",
    description="Get information about a specific Kanji character.",
    parameters={
        "type": "object",
        "properties": {
            "character": {"type": "string", "description": "A single Kanji character"}
        },
        "required": ["character"]
    },
    func=lookup_kanji
))