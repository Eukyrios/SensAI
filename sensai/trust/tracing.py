"""Turn tracing and execution flow aggregation."""
from dataclasses import dataclass
from ..core.events import (
    AgentRunFinished,
    AgentRunStarted,
    Event,
    GuardrailTriggered,
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
    """Collects events during a turn and builds a detailed trace summary."""

    def __init__(self) -> None:
        self.turn_id: str | None = None
        self.agent_traces: list[AgentTrace] = []
        self._current_agent: AgentTrace | None = None
        self.guardrails_triggered: list[str] = []

    def process_event(self, event: Event) -> None:
        match event:
            case TurnStarted(turn_id=tid):
                self.turn_id = tid
                self.agent_traces.clear()
                self.guardrails_triggered.clear()

            case AgentRunStarted(run_id=rid, agent=name):
                self._current_agent = AgentTrace(agent_name=name or "agent", run_id=rid)
                self.agent_traces.append(self._current_agent)

            case Thinking(text=t):
                if self._current_agent:
                    self._current_agent.thought_chars += len(t)

            case Token():
                if self._current_agent:
                    self._current_agent.tokens_count += 1

            case AgentRunFinished() as finished_event:
                if self._current_agent:
                    self._current_agent.ok = getattr(finished_event, "ok", True)
                    
                    # Extraire la valeur réelle d'Ollama si disponible dans AgentRunFinished
                    real_count = (
                        getattr(finished_event, "eval_count", None)
                        or getattr(finished_event, "output_tokens", None)
                        or getattr(finished_event, "completion_tokens", None)
                        or getattr(finished_event, "total_tokens", None)
                    )
                    if real_count is not None and real_count > 0:
                        self._current_agent.tokens_count = real_count
                        
                self._current_agent = None

            case GuardrailTriggered(rule=r):
                self.guardrails_triggered.append(r)

    def format_trace(self) -> str:
        if not self.turn_id:
            return "No trace recorded yet."

        lines = [f"[trace] Turn ID: {self.turn_id}"]
        if not self.agent_traces:
            lines.append("  (no agent execution captured)")
        else:
            for idx, trace in enumerate(self.agent_traces, 1):
                status_str = "OK" if trace.ok else "FAILED"
                lines.append(
                    f"  {idx}. Agent '{trace.agent_name}' ({trace.run_id}) [{status_str}] "
                    f"| {trace.tokens_count} tokens | {trace.thought_chars} reasoning chars"
                )

        if self.guardrails_triggered:
            lines.append(f"  Guardrails triggered: {', '.join(self.guardrails_triggered)}")

        return "\n".join(lines)