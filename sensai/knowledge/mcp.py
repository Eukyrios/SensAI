from typing import Any, Protocol
from sensai.knowledge.tools import Tool, ToolRegistry


class MCPServerProtocol(Protocol):
    name: str

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


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
        description = spec.get("description", f"MCP tool {name} from {server_name}")
        parameters = spec.get("inputSchema") or spec.get("parameters") or {"type": "object", "properties": {}}
        tool_registry.register(
            Tool(
                name=name,
                description=description,
                parameters=parameters,
                func=_make_mcp_caller(server, name, trace_log),
            )
        )
        registered.append(name)

    trace_log.append({
        "event": "mcp_connected",
        "server": server_name,
        "tools": registered,
    })
    return registered