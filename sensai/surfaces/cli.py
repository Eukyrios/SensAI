"""Command-line chat. Run with: python -m sensai"""
import argparse
import sys

from ..core.agent_spec import SpecError
from ..core.config import load_config
from ..llm.ollama import OllamaError
from ..orchestration.chat import Chat

HELP = "Commands: /reset  /quit"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai, a local Japanese tutor")
    parser.add_argument("--model", help="Ollama model for every agent (overrides the agent specs)")
    parser.add_argument("--config", help="path to a config.toml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.model:
            config["agents"]["model_override"] = args.model
        chat = Chat(config)
    except FileNotFoundError as e:
        sys.exit(f"[error] config not found: {e.filename}")
    except SpecError as e:
        sys.exit(f"[error] {e}")

    print(f"Sensai ({chat.spec.model}) — {HELP}")
    while True:
        try:
            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            chat.reset()
            print("History cleared.")
            continue

        try:
            for token in chat.send(msg):
                print(token, end="", flush=True)
            print()
        except OllamaError as e:
            print(f"\n[error] {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[interrupted]")


if __name__ == "__main__":
    main()
