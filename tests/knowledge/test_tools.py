# Tests for the tool registry and the native/prompted tool calling.

import json
from sensai.knowledge.tools import Tool, ToolRegistry, registry, get_tools_for_model

def test_registry_contains_required_tools():
    assert registry.get("dictionary") is not None
    assert registry.get("conjugate") is not None
    assert registry.get("kanji") is not None

def test_tool_execution():
    result = registry.execute("kanji", {"character": "鬱"})
    data = json.loads(result)
    assert data["kanji"] == "鬱"
    assert "onyomi" in data

def test_execute_unknown_tool():
    result = registry.execute("unknown", {})
    assert "Error: Tool 'unknown' not found." in result

def test_native_schema_format():
    tool = registry.get("dictionary")
    schema = tool.to_native_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dictionary"
    assert "word" in schema["function"]["parameters"]["properties"]
    assert "word" in schema["function"]["parameters"]["required"]

def test_prompt_instructions_format():
    instructions = registry.get_prompt_instructions()
    assert "You have access to the following tools:" in instructions
    assert "- dictionary:" in instructions
    assert "- conjugate:" in instructions
    assert "- kanji:" in instructions
    assert "{\"tool\": \"name\", \"args\": {}}" in instructions


class MockOllamaClient:
    def __init__(self, caps):
        self._caps = caps
        
    def capabilities(self, model: str) -> set:
        return self._caps

def test_dual_path_native():
    client = MockOllamaClient({"completion", "tools", "thinking"})
    result = get_tools_for_model(client, "qwen3.5:9b")
    assert result["type"] == "native"
    assert len(result["tools"]) == 3

def test_dual_path_prompted():
    client = MockOllamaClient({"completion"})
    result = get_tools_for_model(client, "old_model")
    assert result["type"] == "prompted"
    assert "You have access to the following tools:" in result["instructions"]
