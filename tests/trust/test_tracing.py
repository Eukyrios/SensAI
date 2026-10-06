"""Tests for per-agent logging and trace formatting."""
from sensai.core.events import (
    AgentRunFinished,
    AgentRunStarted,
    GuardrailTriggered,
    Thinking,
    Token,
    TurnStarted,
)
from sensai.trust.tracing import TurnTracer


def test_turn_tracer_records_and_formats_execution():
    tracer = TurnTracer()

    tracer.process_event(
        TurnStarted(turn_id="turn_123", session_id="sess_1", user_input="Konnichiwa")
    )
    tracer.process_event(
        AgentRunStarted(turn_id="turn_123", run_id="run_1", agent="tutor")
    )
    tracer.process_event(
        Thinking(turn_id="turn_123", run_id="run_1", text="Analyzing grammar...")
    )
    tracer.process_event(Token(turn_id="turn_123", run_id="run_1", text="Konnichiwa"))
    tracer.process_event(
        AgentRunFinished(
            turn_id="turn_123",
            run_id="run_1",
            agent="tutor",
            ok=True,
        )
    )
    tracer.process_event(
        GuardrailTriggered(
            turn_id="turn_123", side="input", rule="no_pii", action="block"
        )
    )

    trace_output = tracer.format_trace()

    assert "Turn ID: turn_123" in trace_output
    assert "Agent 'tutor'" in trace_output
    assert "[OK]" in trace_output
    assert "1 tokens" in trace_output
    assert "Guardrails triggered: no_pii" in trace_output