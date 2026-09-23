"""Outils appelables par le modèle — propriétaire : C
STUB : aucune action réelle."""

TOOLS_SCHEMA: list[dict] = []   # format "tools" de l'API Ollama, à remplir


def run_tool(name: str, args: dict) -> str:
    """Exécute un outil et renvoie son résultat en texte."""
    return f"[stub] outil '{name}' appelé avec {args}"


def needs_confirmation(name: str, config: dict) -> bool:
    return name in config.get("tools", {}).get("require_confirmation", [])