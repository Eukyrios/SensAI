import json
from pathlib import Path
from sensai.knowledge.mcp import register_mcp_server
from sensai.knowledge.rag import BasicRAGIndex, answer_with_rag
from sensai.knowledge.tools import Tool, ToolRegistry, lookup_dictionary

FIXTURE_CORPUS = """# Particles は vs が [N5]
Use は (wa) to mark the topic of a sentence and が (ga) to mark the subject or new focus.

# Conditional たら [N4]
Attach たら (tara) to the past tense form of a verb to express a conditional 'if/when'.

# Advanced Honorifics ざるを得ない [N1]
Used in formal contexts to mean 'cannot help doing'.
"""


class StubMCPServer:
    name = "jlpt-mcp"

    def __init__(self, fail_connect: bool = False, fail_call: bool = False):
        self.fail_connect = fail_connect
        self.fail_call = fail_call

    def list_tools(self) -> list[dict]:
        if self.fail_connect:
            raise ConnectionError("MCP server unreachable")
        return [
            {
                "name": "example_sentence",
                "description": "Fetch an example sentence for a target word.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}},
                    "required": ["word"],
                },
            }
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        if self.fail_call:
            raise RuntimeError("MCP timeout")
        word = arguments.get("word", "")
        return {"word": word, "sentence": f"私は毎日{word}を使います。"}


def _build_fixture_index() -> BasicRAGIndex:
    index = BasicRAGIndex()
    index.ingest_markdown(FIXTURE_CORPUS, source="grammar_guide.md")
    return index


def test_r1_1_grammar_answer_includes_citation_and_trace():
    index = _build_fixture_index()
    result = answer_with_rag("when do I use は vs が?", index)

    assert "[grammar_guide.md § Particles は vs が]" in result.answer
    assert result.citations == ["[grammar_guide.md § Particles は vs が]"]
    trace = result.to_trace_dict()
    assert len(trace["retrieved_chunks"]) >= 1
    assert trace["retrieved_chunks"][0]["source"] == "grammar_guide.md"


def test_r1_1_unanswered_topic_has_no_fake_citation():
    index = _build_fixture_index()
    result = answer_with_rag("how do I bake French croissants?", index)

    assert result.citations == []
    assert result.retrieved_chunks == []
    assert "[" not in result.answer and "§" not in result.answer


def test_r1_2_chunks_carry_jlpt_and_filter_respects_level():
    index = _build_fixture_index()
    levels = {c.jlpt for c in index.chunks}
    assert levels == {"N5", "N4", "N1"}

    n4_results = index.retrieve("formal contexts ざるを得ない conditional たら", max_jlpt="N4", top_k=5)
    assert len(n4_results) >= 1
    assert all(c.jlpt in {"N5", "N4"} for c in n4_results)


def test_r1_2_newly_ingested_document_is_immediately_searchable(tmp_path: Path):
    index = _build_fixture_index()
    new_doc = tmp_path / "counters.md"
    new_doc.write_text("# Flat Objects 枚 [N5]\nUse 枚 (mai) to count flat objects like paper or shirts.", encoding="utf-8")

    index.ingest_file(new_doc)
    result = answer_with_rag("how to count flat objects with 枚?", index)

    assert "[counters.md § Flat Objects 枚]" in result.answer
    assert result.retrieved_chunks[0].source == "counters.md"


def test_t1_2_mcp_tools_appear_alongside_local_tools_and_execute():
    reg = ToolRegistry()
    reg.register(Tool(
        name="dictionary",
        description="Local dictionary",
        parameters={"type": "object", "properties": {"word": {"type": "string"}}},
        func=lookup_dictionary,
    ))
    trace: list[dict] = []
    server = StubMCPServer()

    added = register_mcp_server(reg, server, trace_log=trace)
    assert added == ["example_sentence"]
    assert reg.get("dictionary") is not None
    assert reg.get("example_sentence") is not None

    raw_output = reg.execute("example_sentence", {"word": "日本語"})
    data = json.loads(raw_output)
    assert "日本語" in data["sentence"]
    assert any(e["event"] == "mcp_tool_call" and e["status"] == "ok" for e in trace)


def test_t1_2_unavailable_mcp_server_continues_cleanly_and_logs_trace():
    reg = ToolRegistry()
    reg.register(Tool(
        name="dictionary",
        description="Local dictionary",
        parameters={"type": "object", "properties": {"word": {"type": "string"}}},
        func=lookup_dictionary,
    ))
    trace: list[dict] = []
    offline_server = StubMCPServer(fail_connect=True)

    added = register_mcp_server(reg, offline_server, trace_log=trace)
    assert added == []
    assert reg.get("dictionary") is not None
    assert any(e["event"] == "mcp_unavailable" for e in trace)