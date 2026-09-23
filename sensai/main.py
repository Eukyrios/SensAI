"""CLI — point d'entrée : python -m sensai.main"""
import argparse
import sys

from .config import load_config
from .chat import Agent
from .ollama_client import OllamaError

HELP = "Commandes : /persona <nom>  /reset  /quit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensai — assistant mail local")
    parser.add_argument("--model", help="modèle Ollama (écrase config.yaml)")
    parser.add_argument("--config", help="chemin vers config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except FileNotFoundError as e:
        sys.exit(f"[erreur] config introuvable : {e.filename}")

    agent = Agent(config, model=args.model)
    print(f"Sensai ({agent.client.model}) — {HELP}")

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
            agent.reset()
            print("Historique effacé.")
            continue
        if msg.startswith("/persona"):
            parts = msg.split(maxsplit=1)
            if len(parts) == 2:
                agent.set_persona(parts[1])
                print(f"Persona : {parts[1]}")
            else:
                print(f"Persona actuel : {agent.persona}")
            continue

        try:
            for token in agent.chat(msg):
                print(token, end="", flush=True)
            print()
        except OllamaError as e:
            print(f"\n[erreur] {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[interrompu]")


if __name__ == "__main__":
    main()