# Tests for the per-turn trace.

from sensai.core.events import (
    AgentRunFinished,
    AgentRunStarted,
    GuardrailTriggered,
    Retrieved,
    RouteDecided,
    Thinking,
    Token,
    ToolCall,
    ToolResult,
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


def test_trace_shows_the_routing_decision_of_the_turn():
    tracer = TurnTracer()
    tracer.process_event(TurnStarted(turn_id="t1", session_id="s", user_input="kinou eiga wo mimashita"))
    tracer.process_event(RouteDecided(turn_id="t1", agent="checker", tier="embeddings", confidence=0.654,
                                      reason="closest example 'kare wa sensei desu'"))
    assert "Routed to 'checker' by embeddings (0.65): closest example 'kare wa sensei desu'" in tracer.format_trace()
    tracer.process_event(TurnStarted(turn_id="t2", session_id="s", user_input="hi"))
    assert "Routed" not in tracer.format_trace()


def test_trace_shows_the_retrieval_and_the_tool_calls_of_each_run():
    tracer = TurnTracer()
    for event in [
        TurnStarted(turn_id="t", session_id="s", user_input="q"),
        Retrieved(turn_id="t", run_id="r1", query="q", level="N5", mode="embeddings+model",
                  kept=[{"source": "a.md", "section": "に", "jlpt": "N5", "score": 0.9}],
                  dropped=[{"source": "b.md", "section": "で", "initial_score": 0.5, "rerank_score": 0.1}]),
        AgentRunStarted(turn_id="t", run_id="r1", agent="sensei"),
        ToolCall(turn_id="t", run_id="r1", name="dictionary", args={"word": "猫"}),
        ToolResult(turn_id="t", run_id="r1", name="dictionary", result="x" * 300, ok=False),
        AgentRunFinished(turn_id="t", run_id="r1", agent="sensei", ok=True),
    ]:
        tracer.process_event(event)
    trace = tracer.format_trace()
    assert "rag (embeddings+model, up to N5): 1 kept, 1 dropped" in trace
    assert "kept a.md § に [N5] 0.90" in trace and "dropped b.md § で 0.50 → 0.10" in trace
    assert "tool dictionary(word='猫')" in trace and "→ FAILED: " + "x" * 120 + "…" in trace


def test_trace_shows_the_faithfulness_verdict():
    from sensai.core.events import FaithfulnessChecked
    tracer = TurnTracer()
    tracer.process_event(TurnStarted(turn_id="t", session_id="s", user_input="q"))
    tracer.process_event(FaithfulnessChecked(turn_id="t", faithfulness=0.5, relevance=0.9, supported=1,
                                             flagged=[{"claim": "c", "status": "UNVERIFIED", "reasoning": ""}]))
    assert "Faithfulness: 0.50, relevance 0.90 (1 supported); flagged: 1" in tracer.format_trace()
    tracer.process_event(FaithfulnessChecked(turn_id="t", faithfulness=None, error="down"))
    assert "Faithfulness: not checked (down)" in tracer.format_trace()
