"""Tests for the tool registry and dual path logic."""
import json
from sensai.knowledge.tools import Tool, ToolRegistry, registry

def test_registry_contains_required_tools():
    """Vérifie que les trois outils obligatoires sont bien enregistrés."""
    assert registry.get("dictionary") is not None
    assert registry.get("conjugate") is not None
    assert registry.get("kanji") is not None

def test_tool_execution():
    """Vérifie l'exécution d'un outil et le retour au format JSON stringifié."""
    result = registry.execute("kanji", {"character": "鬱"})
    data = json.loads(result)
    assert data["kanji"] == "鬱"
    assert "onyomi" in data

def test_execute_unknown_tool():
    """Vérifie la gestion d'erreur lors de l'appel d'un outil inexistant."""
    result = registry.execute("unknown", {})
    assert "Error: Tool 'unknown' not found." in result

def test_native_schema_format():
    """Vérifie que le schéma natif correspond au format attendu par Ollama."""
    tool = registry.get("dictionary")
    schema = tool.to_native_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dictionary"
    assert "word" in schema["function"]["parameters"]["properties"]
    assert "word" in schema["function"]["parameters"]["required"]

def test_prompt_instructions_format():
    """Vérifie la génération des instructions pour le fallback 'prompted'."""
    instructions = registry.get_prompt_instructions()
    assert "You have access to the following tools:" in instructions
    assert "- dictionary:" in instructions
    assert "- conjugate:" in instructions
    assert "- kanji:" in instructions
    # Vérifie la présence du format JSON strict exigé pour le fallback
    assert "{\"tool\": \"name\", \"args\": {}}" in instructions