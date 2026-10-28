# Tests for the MCP stdio client against our own MCP server, run as a real subprocess, and for the server.

import io
import json
import sys

import pytest

from sensai.knowledge.mcp import MCPError, StdioMCPServer, register_mcp_server
from sensai.knowledge.mcp_server import MCPServer, serve
from sensai.knowledge.tools import ToolRegistry, registry

FAKE_SERVER = r'''
import json, sys
for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message or "method" not in message:
        continue
    method = message["method"]
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}), flush=True)
        result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "fake"}}
    elif method == "tools/list" and not message["params"]:
        print("not json", flush=True)
        result = {"tools": [{"name": "one", "inputSchema": {"type": "object"}}], "nextCursor": "page2"}
    elif method == "tools/list":
        result = {"tools": [{"name": "two"}]}
    elif message["params"]["name"] == "boom":
        result = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
    elif message["params"]["name"] == "structured":
        result = {"content": [], "structuredContent": {"value": 3}}
    elif message["params"]["name"] == "slow":
        continue
    elif message["params"]["name"] == "quit":
        sys.exit(0)
    else:
        result = {"content": [{"type": "text", "text": "plain words"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
'''


@pytest.fixture
def own_server():
    server = StdioMCPServer("sensai", "{python}", ["-m", "sensai.knowledge.mcp_server"], timeout_s=10)
    yield server
    server.close()


@pytest.fixture
def fake_server(tmp_path):
    script = tmp_path / "fake_mcp.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    server = StdioMCPServer("fake", sys.executable, [str(script)], timeout_s=2)
    yield server
    server.close()


def test_own_server_lists_and_runs_the_own_tools_over_stdio(own_server):
    assert [tool["name"] for tool in own_server.list_tools()] == ["dictionary", "conjugate", "kanji"]
    assert own_server.server_info["name"] == "sensai-japanese"
    assert own_server.call_tool("conjugate", {"verb": "行く", "form": "te"})["result"] == "行って"


def test_a_tool_error_from_the_server_raises(own_server):
    with pytest.raises(MCPError, match="unknown tool"):
        own_server.call_tool("nope", {})
    with pytest.raises(MCPError, match="not a verb"):
        own_server.call_tool("conjugate", {"verb": "猫", "form": "te"})


def test_server_tools_join_the_registry_and_taken_names_get_a_prefix(own_server):
    tools = registry.subset(["dictionary"])
    trace = []
    assert register_mcp_server(tools, own_server, trace) == ["sensai_dictionary", "conjugate", "kanji"]
    text, ok = tools.run("sensai_dictionary", {"word": "猫"})
    assert ok and json.loads(text)["meaning"] == "cat"
    assert tools.get("sensai_dictionary").description.startswith("[MCP sensai]")
    assert trace[-1]["event"] == "mcp_tool_call"


def test_client_follows_pagination_answers_pings_and_reads_results(fake_server):
    assert [tool["name"] for tool in fake_server.list_tools()] == ["one", "two"]
    assert fake_server.call_tool("echo", {}) == "plain words"
    assert fake_server.call_tool("structured", {}) == {"value": 3}
    with pytest.raises(MCPError, match="it broke"):
        fake_server.call_tool("boom", {})


def test_a_silent_server_times_out(fake_server):
    with pytest.raises(MCPError, match="did not answer"):
        fake_server.call_tool("slow", {})


def test_a_server_that_exits_is_reported(fake_server):
    with pytest.raises(MCPError, match="exited"):
        fake_server.call_tool("quit", {})
    with pytest.raises(MCPError):
        fake_server.call_tool("echo", {})


def test_a_missing_command_makes_the_server_unavailable():
    trace = []
    assert register_mcp_server(ToolRegistry(), StdioMCPServer("ghost", "no-such-command-xyz"), trace) == []
    assert trace[-1]["event"] == "mcp_unavailable" and "cannot start" in trace[-1]["reason"]


def test_close_twice_is_harmless(own_server):
    own_server.list_tools()
    own_server.close()
    own_server.close()
    with pytest.raises(MCPError, match="not running"):
        own_server._send({})


def handle(message):
    return MCPServer(registry).handle(message)


def test_server_negotiates_the_protocol_version():
    asked = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert asked["result"]["protocolVersion"] == "2024-11-05"
    unknown = handle({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999"}})
    assert unknown["result"]["protocolVersion"] == "2025-06-18"


def test_server_answers_ping_ignores_notifications_and_rejects_unknown_methods():
    assert handle({"jsonrpc": "2.0", "id": 1, "method": "ping"})["result"] == {}
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handle({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})["error"]["code"] == -32601


def test_serve_reads_lines_and_reports_parse_errors():
    requests = "\n".join([json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}), "", "{bad", "[1]"])
    out = io.StringIO()
    serve(io.StringIO(requests + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(replies[0]["result"]["tools"]) == 3
    assert replies[1]["error"]["code"] == -32700 and len(replies) == 2


NOISY_SERVER = r'''
import json, sys, time
out = sys.stdout.buffer
for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message or "method" not in message:
        continue
    if message["method"] == "initialize":
        version = sys.argv[1]
        result = {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": "noisy"}}
    elif message["params"].get("name") == "chatty":
        for _ in range(20):
            out.write(b'{"jsonrpc": "2.0", "method": "notifications/progress"}\n'); out.flush(); time.sleep(0.1)
        continue
    else:
        out.write(b"\xff\xfe not utf-8\n"); out.flush()
        result = {"content": [{"type": "text", "text": "still here"}]}
    out.write((json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n").encode()); out.flush()
'''


def noisy_server(tmp_path, version="2025-06-18", timeout_s=0.5):
    script = tmp_path / "noisy.py"
    script.write_text(NOISY_SERVER, encoding="utf-8")
    return StdioMCPServer("noisy", sys.executable, [str(script), version], timeout_s=timeout_s)


def test_invalid_utf8_from_the_server_does_not_kill_the_reader(tmp_path):
    server = noisy_server(tmp_path, timeout_s=5)
    try:
        assert server.call_tool("echo", {}) == "still here"
        assert server.call_tool("echo", {}) == "still here"
    finally:
        server.close()


def test_notifications_do_not_extend_the_request_timeout(tmp_path):
    import time
    server = noisy_server(tmp_path)
    try:
        server.start()
        started = time.monotonic()
        with pytest.raises(MCPError, match="did not answer"):
            server.call_tool("chatty", {})
        assert time.monotonic() - started < 1.5
    finally:
        server.close()


def test_an_unsupported_protocol_version_is_refused(tmp_path):
    server = noisy_server(tmp_path, version="1999-01-01", timeout_s=5)
    with pytest.raises(MCPError, match="speaks MCP 1999-01-01"):
        server.start()
    assert server.process is None


def test_a_tool_error_is_not_reported_as_an_unavailable_server(own_server):
    tools = ToolRegistry()
    trace = []
    register_mcp_server(tools, own_server, trace)
    text, ok = tools.run("conjugate", {"verb": "猫", "form": "te"})
    assert not ok and "'conjugate' failed" in text and "unavailable" not in text
    assert trace[-1]["status"] == "error"


def test_the_server_gets_a_minimal_environment(monkeypatch):
    monkeypatch.setenv("SECRET_API_KEY", "do-not-leak")
    server = StdioMCPServer("x", "cmd", env={"ONLY": 1})
    assert "SECRET_API_KEY" not in server.env and server.env["ONLY"] == "1" and "PATH" in server.env


class NamedServer:
    name = "my server"

    def __init__(self, *names):
        self.names = names

    def list_tools(self):
        return [{"name": name} for name in self.names]

    def call_tool(self, name, arguments):
        return name


def test_tool_names_are_made_safe_and_unique():
    tools = registry.subset(["kanji"])
    names = register_mcp_server(tools, NamedServer("get/time now", "kanji", "kanji", "x" * 80))
    assert names == ["get_time_now", "my_server_kanji", "my_server_kanji_2", "x" * 64]


def test_server_rejects_bad_params_without_dying():
    assert handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "x"})["error"]["code"] == -32602
    assert handle({"jsonrpc": "2.0", "id": 2, "result": {}}) is None


def test_server_reports_an_internal_error():
    class Broken(ToolRegistry):
        def names(self):
            raise RuntimeError("boom")
    reply = MCPServer(Broken()).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert reply["error"]["code"] == -32603 and "boom" in reply["error"]["message"]
