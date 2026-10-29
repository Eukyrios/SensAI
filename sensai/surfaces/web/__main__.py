# Entry point for `python -m sensai.surfaces.web`: serves the web backend with uvicorn.

import argparse

import uvicorn

from ...core.config import load_config
from .app import create_app


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config, cli={"web": {"host": args.host, "port": args.port}})
    uvicorn.run(create_app(config), host=config["web"]["host"], port=config["web"]["port"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai-web", description="Sensai web backend")
    parser.add_argument("--config", help="path to a config.toml")
    parser.add_argument("--host", help="address to listen on (default: [web] host in config.toml)")
    parser.add_argument("--port", type=int, help="port to listen on (default: [web] port in config.toml)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
