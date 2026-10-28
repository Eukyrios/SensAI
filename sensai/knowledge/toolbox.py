from typing import Any
from ..core.agent_spec import AgentSpec
from .mcp import StdioMCPServer, register_mcp_server
from .retrieval import Retrieval, Retriever
from .tools import ToolRegistry, own_tools

MCP_PREFIX = "mcp:"

class Knowledge:

    def __init__(self, config: dict, client: Any):
        self.config = config
        self.client = client
        self.registry = ToolRegistry()
        for tool in own_tools():
            self.registry.register(tool)
        self.servers: list[StdioMCPServer] = []
        self.server_tools: dict[str, list[str]] = {}
        self.unavailable: dict[str, str] = {}
        self.trace_log: list[dict] = []
        self.retriever = Retriever(config["rag"], client) if config["rag"]["enabled"] else None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for settings in self.enabled_servers():
            self._connect(settings)

    def enabled_servers(self) -> list[dict]:
        if not self.config["tools"]["enabled"] or not self.config["mcp"]["enabled"]:
            return []
        return [settings for settings in self.config["mcp"]["servers"] if settings.get("enabled", True)]

    def _connect(self, settings: dict) -> None:
        name = settings["name"]
        server = StdioMCPServer(name, settings["command"], settings.get("args", ()), settings.get("env"),
                                settings.get("timeout_s", 20))
        try:
            registered = register_mcp_server(self.registry, server, self.trace_log)
        except BaseException:
            server.close()
            raise
        if registered:
            self.servers.append(server)
            self.server_tools[name] = registered
        else:
            server.close()
            self.unavailable[name] = self.trace_log[-1].get("reason", "the server lists no tools")

    def tools_for(self, spec: AgentSpec) -> ToolRegistry | None:
        if not spec.tools or not self.config["tools"]["enabled"]:
            return None
        if any(name.startswith(MCP_PREFIX) for name in spec.tools):
            self.start()
        names: list[str] = []
        for name in spec.tools:
            names.extend(self.server_tools.get(name[len(MCP_PREFIX):], []) if name.startswith(MCP_PREFIX)
                         else [name])
        chosen = self.registry.subset(names)
        return chosen if len(chosen) else None

    def retrieve(self, question: str, profile_text: str | None, model: str) -> Retrieval | None:
        if self.retriever is None:
            return None
        return self.retriever.retrieve(question, profile_text, model)

    def close(self) -> None:
        for server in self.servers:
            server.close()
        self.servers.clear()
