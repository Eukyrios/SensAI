"""Command-line chat interface and CLI renderer."""
import argparse
import asyncio
import sys
from typing import AsyncGenerator

from ..core.agent_spec import SpecError
from ..core.config import load_config
from ..core.events import Error, Event, GuardrailTriggered, Token, TurnFinished
from ..llm.ollama import OllamaError
from ..orchestration.session import Session
from ..orchestration.turn import run_turn
from ..trust.logging import JsonlLogger

COMMANDS = {
    "/help": ("/help", "list the commands"),
    "/model": ("/model [name]", "show the current and installed models, or switch model (history kept)"),
    "/reset": ("/reset", "clear the conversation history"),
    "/quit": ("/quit", "exit (Ctrl-D works too)"),
}
HELP = "Commands: " + "  ".join(usage for usage, _ in COMMANDS.values())


class CLIRenderer:
    """Consumes events from an Event stream and renders them to the CLI."""

    async def render_stream(self, stream: AsyncGenerator[Event, None]) -> None:
        async for event in stream:
            match event:
                case Token(text=t):
                    print(t, end="", flush=True)
                case GuardrailTriggered(reason=r):
                    print(f"\n[guardrail] {r}", file=sys.stderr)
                case Error(message=m):
                    print(f"\n[error] {m}", file=sys.stderr)
                case TurnFinished():
                    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai, a local Japanese tutor")
    parser.add_argument("--model", help="Ollama model for every agent (overrides the agent specs)")
    parser.add_argument("--config", help="path to a config.toml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, cli={"agents": {"model_override": args.model}})
        session = Session(config)
        session.client.require_model(session.spec.model)
    except FileNotFoundError as e:
        sys.exit(f"[error] config not found: {e.filename}")
    except (SpecError, OllamaError) as e:
        sys.exit(f"[error] {e}")

    logger = JsonlLogger("logs/events.jsonl")
    renderer = CLIRenderer()

    print(f"Sensai ({session.spec.model}) — {HELP}")
    while True:
        try:
            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not msg:
            continue
        if msg.startswith("/"):
            if not run_command(session, msg):
                break
            continue

        try:
            # Multi-consumer flow: turn stream -> logger pass-through -> CLI renderer
            event_stream = run_turn(session, msg)
            logged_stream = logger.log_stream(event_stream)
            asyncio.run(renderer.render_stream(logged_stream))
        except KeyboardInterrupt:
            print("\n[interrupted]")


def run_command(session: Session, line: str) -> bool:
    """Handle a /command. Returns False when the chat should stop."""
    name, _, arg = line.partition(" ")
    arg = arg.strip()
    if name == "/quit":
        return False
    if name == "/help":
        width = max(len(usage) for usage, _ in COMMANDS.values())
        print("\n".join(f"  {usage:<{width}}  {text}" for usage, text in COMMANDS.values()))
    elif name == "/reset":
        session.reset()
        print("History cleared.")
    elif name == "/model":
        try:
            if arg:
                session.set_model(arg)
                print(f"Model: {arg} (history kept)")
            else:
                installed = ", ".join(session.client.list_models()) or "none"
                print(f"Model: {session.spec.model}\nInstalled: {installed}")
        except OllamaError as e:
            print(f"[error] {e}", file=sys.stderr)
    else:
        print(f"Unknown command {name}. Type /help for the list.")
    return True


if __name__ == "__main__":
    main()