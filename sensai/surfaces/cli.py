"""Command-line chat. Run with: python -m sensai"""
import argparse
import asyncio
import sys

from ..core.agent_spec import SpecError
from ..core.config import load_config
from ..core.events import Error, Token, TurnFinished
from ..llm.ollama import OllamaError
from ..orchestration.session import Session
from ..orchestration.turn import run_turn

COMMANDS = {
    "/help": ("/help", "list the commands"),
    "/model": ("/model [name]", "show the current and installed models, or switch model (history kept)"),
    "/reset": ("/reset", "clear the conversation history"),
    "/quit": ("/quit", "exit (Ctrl-D works too)"),
}
HELP = "Commands: " + "  ".join(usage for usage, _ in COMMANDS.values())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai, a local Japanese tutor")
    parser.add_argument("--model", help="Ollama model for every agent (overrides the agent specs)")
    parser.add_argument("--config", help="path to a config.toml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, cli={"agents": {"model_override": args.model}})
        session = Session(config)
        session.client.require_model(session.spec.model)     # fail now, not on the first message
    except FileNotFoundError as e:
        sys.exit(f"[error] config not found: {e.filename}")
    except (SpecError, OllamaError) as e:
        sys.exit(f"[error] {e}")

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
            asyncio.run(render_turn(session, msg))
        except KeyboardInterrupt:
            print("\n[interrupted]")


async def render_turn(session: Session, text: str) -> None:
    """Print one turn from its events. Minimal renderer until the CLI renderer (#12) replaces it."""
    async for event in run_turn(session, text):
        match event:
            case Token(text=t):
                print(t, end="", flush=True)
            case Error(message=m):
                print(f"\n[error] {m}", file=sys.stderr)
            case TurnFinished():
                print()


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
