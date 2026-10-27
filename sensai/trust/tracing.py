# Per-turn trace of the routing decision and the agent runs, for /trace.

from dataclasses import dataclass
from ..core.events import (
    AgentRunFinished,
    AgentRunStarted,
    Event,
    GuardrailTriggered,
    RouteDecided,
    Thinking,
    Token,
    TurnStarted,
)


@dataclass
class AgentTrace:
    agent_name: str
    run_id: str
    tokens_count: int = 0
    thought_chars: int = 0
    ok: bool = True


class TurnTracer:

    def __init__(self) -> None:
        self.turn_id: str | None = None
        self.agent_traces: list[AgentTrace] = []
        self._current_agent: AgentTrace | None = None
        self.guardrails_triggered: list[str] = []
        self.route: RouteDecided | None = None

    def process_event(self, event: Event) -> None:
        match event:
            case TurnStarted(turn_id=turn_id):
                self._start_turn(turn_id)
            case AgentRunStarted(run_id=run_id, agent=agent):
                self._start_agent_run(run_id, agent)
            case Thinking(text=text):
                self._count_reasoning(text)
            case Token():
                self._count_token()
            case AgentRunFinished(ok=ok):
                self._finish_agent_run(ok)
            case RouteDecided() as route:
                self.route = route
            case GuardrailTriggered(rule=rule):
                self.guardrails_triggered.append(rule)

    def format_trace(self) -> str:
        if not self.turn_id:
            return "No trace recorded yet."
        lines = [f"[trace] Turn ID: {self.turn_id}"]
        if self.route is not None:
            lines.append(_format_route(self.route))
        if not self.agent_traces:
            lines.append("  (no agent execution captured)")
        lines.extend(_format_agent_run(number, trace) for number, trace in enumerate(self.agent_traces, 1))
        if self.guardrails_triggered:
            lines.append(f"  Guardrails triggered: {', '.join(self.guardrails_triggered)}")
        return "\n".join(lines)

    def _start_turn(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.agent_traces.clear()
        self.guardrails_triggered.clear()
        self.route = None

    def _start_agent_run(self, run_id: str, agent: str) -> None:
        self._current_agent = AgentTrace(agent_name=agent or "agent", run_id=run_id)
        self.agent_traces.append(self._current_agent)

    def _count_reasoning(self, text: str) -> None:
        if self._current_agent:
            self._current_agent.thought_chars += len(text)

    def _count_token(self) -> None:
        if self._current_agent:
            self._current_agent.tokens_count += 1

    def _finish_agent_run(self, ok: bool) -> None:
        if self._current_agent:
            self._current_agent.ok = ok
        self._current_agent = None


def _format_route(route: RouteDecided) -> str:
    return f"  Routed to '{route.agent}' by {route.tier} ({route.confidence:.2f}): {route.reason}"


def _format_agent_run(number: int, trace: AgentTrace) -> str:
    status = "OK" if trace.ok else "FAILED"
    return (f"  {number}. Agent '{trace.agent_name}' ({trace.run_id}) [{status}] "
            f"| {trace.tokens_count} tokens | {trace.thought_chars} reasoning chars")
