import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Any, Protocol
from sensai.knowledge.tools import Tool, ToolRegistry

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
CLIENT_INFO = {"name": "sensai", "version": "1.0"}
INHERITED_ENV = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "UV_CACHE_DIR")
TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]")
MAX_TOOL_NAME = 64

class MCPError(RuntimeError):
    """The server could not be reached: it did not start, answer in time, or it exited."""

class MCPToolError(MCPError):
    """The server answered, but the tool reported an error (isError) or rejected the request."""


class MCPServerProtocol(Protocol):
    name: str

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class StdioMCPServer:

    def __init__(self, name: str, command: str, args: list[str] | tuple[str, ...] = (),
                 env: dict[str, str] | None = None, timeout_s: float = 20, cwd: str | None = None):
        self.name = name
        self.command = [sys.executable if command == "{python}" else command, *args]
        inherited = {name: os.environ[name] for name in INHERITED_ENV if name in os.environ}
        self.env = {**inherited, **{key: str(value) for key, value in (env or {}).items()}}
        self.timeout_s = timeout_s
        self.cwd = cwd
        self.process: subprocess.Popen | None = None
        self.server_info: dict = {}
        self._ids = itertools.count(1)
        self._lines: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None:
            return
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, env=self.env, cwd=self.cwd,
                                            text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as error:
            raise MCPError(f"cannot start {self.command[0]}: {error}") from None
        threading.Thread(target=self._read_stdout, name=f"mcp-{self.name}", daemon=True).start()
        result = self._request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                              "clientInfo": CLIENT_INFO})
        version = result.get("protocolVersion")
        if version not in SUPPORTED_VERSIONS:
            self.close()
            raise MCPError(f"{self.name} speaks MCP {version}, not one of {', '.join(SUPPORTED_VERSIONS)}")
        self.server_info = result.get("serverInfo", {})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def list_tools(self) -> list[dict[str, Any]]:
        self.start()
        tools: list[dict] = []
        cursor = None
        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.start()
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(item.get("text", "") for item in result.get("content", []) if item.get("type") == "text")
        if result.get("isError"):
            raise MCPToolError(text or f"{name} failed")
        if "structuredContent" in result:
            return result["structuredContent"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def close(self) -> None:
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=1)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    def _request(self, method: str, params: dict) -> dict:
        deadline = time.monotonic() + self.timeout_s
        if not self._lock.acquire(timeout=self.timeout_s):
            raise MCPError(f"{self.name} is still busy with an earlier call")
        try:
            request_id = next(self._ids)
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            try:
                message = self._response_to(request_id, deadline)
            except MCPError:
                self._cancel(request_id)
                raise
        finally:
            self._lock.release()
        if "error" in message:
            error = message["error"]
            raise MCPToolError(f"{method}: {error.get('message', error) if isinstance(error, dict) else error}")
        return message.get("result", {})

    def _response_to(self, request_id: int, deadline: float) -> dict:
        while True:
            message = self._next_message(deadline)
            if message.get("id") == request_id and "method" not in message:
                return message
            if "method" in message and "id" in message:
                self._answer_server_request(message)

    def _cancel(self, request_id: int) -> None:
        try:
            self._send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                        "params": {"requestId": request_id, "reason": "timeout"}})
        except MCPError:
            pass

    def _answer_server_request(self, message: dict) -> None:
        if message["method"] == "ping":
            self._send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
        else:
            self._send({"jsonrpc": "2.0", "id": message["id"],
                        "error": {"code": -32601, "message": f"method not supported: {message['method']}"}})

    def _send(self, message: dict) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPError(f"{self.name} is not running")
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise MCPError(f"{self.name} closed its input") from None

    def _next_message(self, deadline: float) -> dict:
        try:
            line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise MCPError(f"{self.name} did not answer within {self.timeout_s}s") from None
        if line is None:
            raise MCPError(f"{self.name} exited")
        return line

    def _read_stdout(self) -> None:
        process = self.process
        assert process is not None and process.stdout is not None
        try:
            for raw in process.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    self._lines.put(message)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)


def _make_mcp_caller(server: Any, tool_name: str, trace_log: list[dict[str, Any]]):
    def _caller(**kwargs: Any) -> Any:
        try:
            result = server.call_tool(tool_name, kwargs)
            trace_log.append({
                "event": "mcp_tool_call",
                "server": getattr(server, "name", "mcp"),
                "tool": tool_name,
                "args": kwargs,
                "status": "ok",
            })
            return result
        except MCPToolError as exc:
            trace_log.append({
                "event": "mcp_tool_error",
                "server": getattr(server, "name", "mcp"),
                "tool": tool_name,
                "error": str(exc),
                "status": "error",
            })
            return {"error": f"'{tool_name}' failed: {exc}"}
        except Exception as exc:
            trace_log.append({
                "event": "mcp_tool_error",
                "server": getattr(server, "name", "mcp"),
                "tool": tool_name,
                "error": str(exc),
                "status": "unavailable",
            })
            return {"error": f"MCP server unavailable during '{tool_name}': {exc}"}

    return _caller


def register_mcp_server(
    tool_registry: ToolRegistry,
    server: Any,
    trace_log: list[dict[str, Any]] | None = None,
) -> list[str]:
    if trace_log is None:
        trace_log = []

    server_name = getattr(server, "name", "external_mcp")
    try:
        remote_tools = server.list_tools()
    except Exception as exc:
        trace_log.append({
            "event": "mcp_unavailable",
            "server": server_name,
            "reason": str(exc),
        })
        return []

    registered: list[str] = []
    for spec in remote_tools:
        name = spec["name"]
        local_name = _local_name(tool_registry, server_name, name)
        description = spec.get("description", f"MCP tool {name} from {server_name}")
        parameters = spec.get("inputSchema") or spec.get("parameters") or {"type": "object", "properties": {}}
        tool_registry.register(
            Tool(
                name=local_name,
                description=f"[MCP {server_name}] {description}",
                parameters=parameters,
                func=_make_mcp_caller(server, name, trace_log),
            )
        )
        registered.append(local_name)

    trace_log.append({
        "event": "mcp_connected",
        "server": server_name,
        "tools": registered,
    })
    return registered


def _local_name(tool_registry: ToolRegistry, server_name: str, name: str) -> str:
    base = TOOL_NAME.sub("_", name)[:MAX_TOOL_NAME]
    if tool_registry.get(base) is None:
        return base
    candidate = TOOL_NAME.sub("_", f"{server_name}_{name}")[:MAX_TOOL_NAME]
    number = 2
    while tool_registry.get(candidate) is not None:
        suffix = f"_{number}"
        candidate = candidate[:MAX_TOOL_NAME - len(suffix)] + suffix
        number += 1
    return candidate
