"""Demo script wiring both CLI renderer and JsonlLogger over a stubbed event stream."""
import asyncio
from pathlib import Path

from ..core.events import AgentRunFinished, AgentRunStarted, Token, TurnFinished, TurnStarted
from ..trust.logging import JsonlLogger
from .cli import CLIRenderer


async def stub_event_stream():
    """Simulates an event stream for a single turn."""
    yield TurnStarted(session_id="test-session", turn_id="turn-123", user_input="hello")
    yield AgentRunStarted(run_id="run-1", agent="sensei")
    yield Token(turn_id="turn-123", run_id="run-1", text="Konnichiwa ")
    yield Token(turn_id="turn-123", run_id="run-1", text="means ")
    yield Token(turn_id="turn-123", run_id="run-1", text="Hello!")
    yield AgentRunFinished(run_id="run-1", status="success", duration_ms=120)
    yield TurnFinished(turn_id="turn-123", model="qwen3.5:9b")


async def run_demo():
    log_file = Path("logs/demo_events.jsonl")
    if log_file.exists():
        log_file.unlink()

    logger = JsonlLogger(log_file)
    renderer = CLIRenderer()

    print("=== Demo: Stream going through Logger and CLI Renderer ===")
    stream = stub_event_stream()
    logged_stream = logger.log_stream(stream)
    await renderer.render_stream(logged_stream)

    print("\n=== Log file content written by JsonlLogger ===")
    print(log_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(run_demo())