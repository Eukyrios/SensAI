import json
import sys
from typing import Any, TextIO

from sensai.knowledge.tools import ToolRegistry, registry as own_registry

SERVER_INFO = {"name": "sensai-japanese", "version": "1.0"}
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

class MCPServer:

    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    def handle(self, message: dict) -> dict | None:
        method = message.get("method")
        if "id" not in message or method is None:
            return None
        params = message.get("params") or {}
        try:
            if not isinstance(params, dict):
                raise _RPCError(-32602, "params must be an object")
            result = self._dispatch(method, params)
        except _RPCError as error:
            return _error(message["id"], error.code, str(error))
        except Exception as error:
            return _error(message["id"], -32603, f"internal error: {error}")
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}

    def _dispatch(self, method: str | None, params: dict) -> dict:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
            return {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [_tool_spec(self.tools.get(name)) for name in self.tools.names()]}
        if method == "tools/call":
            return self._call(params)
        raise _RPCError(-32601, f"method not found: {method}")

    def _call(self, params: dict) -> dict:
        name = params.get("name", "")
        if self.tools.get(name) is None:
            raise _RPCError(-32602, f"unknown tool: {name}")
        text, ok = self.tools.run(name, params.get("arguments") or {})
        return {"content": [{"type": "text", "text": text}], "isError": not ok}

def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

class _RPCError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code

def _tool_spec(tool: Any) -> dict:
    return {"name": tool.name, "description": tool.description, "inputSchema": tool.parameters}

def serve(stdin: TextIO, stdout: TextIO, server: MCPServer | None = None) -> None:
    server = server or MCPServer(own_registry)
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            response = server.handle(message) if isinstance(message, dict) else None
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()

if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    serve(sys.stdin, sys.stdout)
