"""Command-line chat interface and CLI renderer. Run with: python -m sensai"""
import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from ..core.agent_spec import SpecError
from ..core.config import load_config, resolve
from ..core.errors import ErrorKind
from ..core.events import (AgentRunStarted, Error, Event, GuardrailTriggered, StructuredOutput,
                           Thinking, Token, TurnFinished)
from ..llm.ollama import OllamaError
from ..memory.repository import SessionNotFound, open_repository
from ..orchestration.session import Session
from ..orchestration.turn import run_turn
from ..trust.logging import JsonlLogger
from ..trust.metrics import compute_turn_metrics
from ..trust.tracing import TurnTracer

COMMANDS = {
    "/help": ("/help", "list the commands"),
    "/model": ("/model [name]", "show the current and installed models, or switch model (history kept)"),
    "/think": ("/think [on|off]", "show whether the model reasons before answering, or switch it"),
    "/metrics": ("/metrics", "show token usage, speed, and cost estimation for the last turn"),
    "/trace": ("/trace", "show the agent runs of the last turn"),
    "/reset": ("/reset", "clear the history and start a new session (the old one stays stored)"),
    "/quit": ("/quit", "exit (Ctrl-D works too)"),
}
HELP = "Commands: " + "  ".join(usage for usage, _ in COMMANDS.values())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai, a local Japanese tutor")
    parser.add_argument("--model", help="Ollama model for every agent (overrides the agent specs)")
    parser.add_argument("--agent", help="agent to talk to, from agents/<name>.toml (default: sensei)")
    parser.add_argument("--config", help="path to a config.toml")
    parser.add_argument("--session", metavar="ID", help="resume a stored session (see --list-sessions)")
    parser.add_argument("--list-sessions", action="store_true", help="list the stored sessions and exit")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, cli={"agents": {"model_override": args.model,
                                                          "default": args.agent}})
        if args.list_sessions:
            list_sessions(config)
            return
        session = Session.resume(config, args.session) if args.session else Session(config)
        session.client.require_model(session.spec.model)     # fail now, not on the first message
    except FileNotFoundError as e:
        sys.exit(f"[error] config not found: {e.filename}")
    except SessionNotFound:
        sys.exit(f"[error] session not found: {args.session} (see --list-sessions)")
    except (SpecError, OllamaError) as e:
        sys.exit(f"[error] {e}")

    logger = JsonlLogger(resolve(config["logging"]["path"]))     # [logging] path in config.toml
    print(f"Sensai ({session.spec.model}) — type /help for commands")
    if args.session:
        print(f"Resumed session {session.id} ({len(session.history)} messages)")
    if sys.stdin.isatty() and sys.stdout.isatty():
        asyncio.run(live_chat(session, logger))
    else:
        plain_chat(session, logger)


def list_sessions(config: dict) -> None:
    sessions = open_repository(config).list_sessions()
    if not sessions:
        print("No stored sessions.")
    for row in sessions:
        print(f"{row['id']}  {row['created_at']}  {row['model']}  {row['title'] or '-'}")


Status = Callable[[str | None], None]


def plain_chat(session: Session, logger: JsonlLogger | None = None) -> None:
    """Line by line, for pipes and scripts: read a message, answer it, repeat."""
    renderer = CLIRenderer(PlainOutput())     # one per chat: /metrics and /trace read its last turn
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if not asyncio.run(handle_line(session, line, renderer, logger)):
                return
        except KeyboardInterrupt:
            print("\n[interrupted]")


async def live_chat(session: Session, logger: JsonlLogger | None = None) -> None:
    """In a terminal: the reply streams above a live input line, so the next question can be typed
    while the model is answering. Enter queues it; it runs when the current reply ends.
    Esc stops the reply being generated (the chat goes on); Ctrl-D quits; Ctrl-C does nothing.
    """
    pending: asyncio.Queue[str | None] = asyncio.Queue()
    state: dict = {"status": None, "turn": None, "quit": False}
    out = LiveOutput()

    def toolbar() -> str:
        if state["status"]:
            parts = [f"{state['status']}  ·  Esc: stop the reply"]
        else:
            parts = ["Enter: send  ·  Esc: stop a reply  ·  Ctrl-D: quit  ·  /help"]
        if pending.qsize():
            parts.append(f"{pending.qsize()} queued")
        return "  ·  ".join(parts)

    def message() -> str:
        return ANSI(f"{out.partial}\n> " if out.partial else "> ")

    replying = Condition(lambda: state["turn"] is not None and not state["turn"].done())
    keys = KeyBindings()

    @keys.add("escape", filter=replying, eager=True)
    def _stop_reply(event) -> None:
        if state["turn"]:
            state["turn"].cancel()

    @keys.add("c-c")
    def _ignore(event) -> None:
        pass

    prompt = PromptSession(erase_when_done=True, bottom_toolbar=toolbar, key_bindings=keys)
    prompt.app.ttimeoutlen = 0.05

    def redraw() -> None:
        if prompt.app.is_running:
            prompt.app.invalidate()

    def set_status(status: str | None) -> None:
        state["status"] = status
        redraw()

    out.on_change = redraw
    renderer = CLIRenderer(out, set_status)     # one per chat: /metrics and /trace read its last turn

    async def read_lines() -> None:
        while True:
            try:
                line = await prompt.prompt_async(message)
            except EOFError:
                state["quit"] = True
                turn = state["turn"]
                if turn is not None and not turn.done():
                    turn.cancel()
                await pending.put(None)
                return
            await pending.put(line)

    with patch_stdout(raw=True):
        reader = asyncio.create_task(read_lines())
        try:
            while not state["quit"] and (line := await pending.get()) is not None:
                if not line.strip():
                    continue
                print(f"\n> {line.strip()}")
                state["turn"] = asyncio.create_task(handle_line(session, line, renderer, logger))
                try:
                    keep_going = await state["turn"]
                except asyncio.CancelledError:
                    out.end_line()
                    print("[interrupted]")
                    keep_going = True
                finally:
                    state["turn"] = None
                    set_status(None)
                if not keep_going:
                    return
        finally:
            reader.cancel()


class PlainOutput:
    """Streams text straight to stdout (pipes, scripts, tests)."""

    def __init__(self) -> None:
        self.mid_line = False

    def write(self, text: str) -> None:
        print(text, end="", flush=True)
        self.mid_line = not text.endswith("\n")

    def end_line(self) -> None:
        if self.mid_line:
            print()
            self.mid_line = False


class LiveOutput:
    """Prints finished lines; keeps the unfinished one in `partial` for the live prompt to draw."""

    def __init__(self) -> None:
        self.partial = ""
        self.on_change: Callable[[], None] = lambda: None

    def write(self, text: str) -> None:
        *finished, self.partial = (self.partial + text).split("\n")
        for line in finished:
            print(line)
        self.on_change()

    def end_line(self) -> None:
        if self.partial:
            print(self.partial)
            self.partial = ""
            self.on_change()


Output = PlainOutput | LiveOutput


async def handle_line(session: Session, line: str, renderer: "CLIRenderer",
                      logger: JsonlLogger | None = None) -> bool:
    """One input line: a /command or a message for the model. Returns False when the chat should stop."""
    line = line.strip()
    if not line:
        return True
    if line.startswith("/"):
        return run_command(session, line, renderer)
    await render_turn(session, line, renderer, logger)
    return True


async def render_turn(session: Session, text: str, renderer: "CLIRenderer",
                      logger: JsonlLogger | None = None) -> None:
    """One turn: run_turn -> (logger, which writes each event and passes it on) -> CLI renderer."""
    stream = run_turn(session, text)
    if logger is not None:
        stream = logger.log_stream(stream)
    await renderer.render_stream(stream)


class CLIRenderer:
    """Consumes events from an Event stream and renders them to the CLI."""

    def __init__(self, out: Output | None = None, status: Status | None = None) -> None:
        self.out = out or PlainOutput()
        self.status = status or (lambda _: None)
        self.tracer = TurnTracer()

    async def render_stream(self, stream: AsyncIterator[Event]) -> None:
        out, status = self.out, self.status
        reasoning, answering = False, False
        try:
            async for event in stream:
                self.tracer.process_event(event)
                match event:
                    case AgentRunStarted():
                        status("thinking…")
                    case Thinking(text=t):
                        if not reasoning:
                            out.write(dim("thinking:") + "\n")
                            reasoning = True
                        out.write(dim_lines(t))
                    case Token(text=t):
                        status("replying…")
                        if reasoning and not answering:
                            out.end_line()
                            out.write("\n")
                        answering = True
                        out.write(t)
                    case GuardrailTriggered(side=side, rule=rule, action=action):
                        out.end_line()
                        print(f"[guardrail] {side} {action}: {rule}", file=sys.stderr)
                    case Error(kind=k, message=m):
                        out.end_line()
                        self._render_error(k, m)
                    case StructuredOutput(data=data, attempts=attempts):
                        out.end_line()
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                        if attempts > 1:
                            print(dim("(repaired: the first reply didn't match the schema)"))
                    case TurnFinished() as finished:
                        status(None)
                        out.end_line()
                        if finished.eval_count:
                            metrics = compute_turn_metrics(finished)
                            self.last_metrics = metrics
                            print(dim(usage(finished)))
        except asyncio.CancelledError:
            print("\n[interrupted]", file=sys.stderr)
            raise

    def _render_error(self, kind: ErrorKind | str, message: str) -> None:
        """Format and print user-friendly error messages based on error kind."""
        if kind == "model_missing":
            print(f"\n[error] Model not found. Run:\n  {message}", file=sys.stderr)
        elif kind == "connection":
            print(f"\n[error] Cannot reach Ollama. Start it with:\n  ollama serve", file=sys.stderr)
        elif kind == "timeout":
            print(f"\n[error] Request timed out: {message}", file=sys.stderr)
        else:
            print(f"\n[error] {message}", file=sys.stderr)


def usage(finished: TurnFinished) -> str:
    """e.g. "102 prompt + 11 reply tokens · first token 0.3s · 1.2s" """
    parts = [f"{finished.prompt_eval_count} prompt + {finished.eval_count} reply tokens"]
    if finished.ttft_ms is not None:
        parts.append(f"first token {finished.ttft_ms / 1000:.1f}s")
    parts.append(f"{finished.total_duration_ns / 1e9:.1f}s")
    return " · ".join(parts)


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if sys.stdout.isatty() else text


def dim_lines(text: str) -> str:
    return "\n".join(dim(part) if part else part for part in text.split("\n"))


def run_command(session: Session, line: str, renderer: CLIRenderer | None = None) -> bool:
    """Handle a /command. Returns False when the chat should stop."""
    name, _, arg = line.partition(" ")
    arg = arg.strip()
    if name == "/quit":
        return False
    if name == "/help":
        width = max(len(u) for u, _ in COMMANDS.values())
        print("\n".join(f"  {u:<{width}}  {t}" for u, t in COMMANDS.values()))
    elif name == "/reset":
        session.reset()
        print("History cleared.")
    elif name == "/metrics":
        if renderer and hasattr(renderer, "last_metrics") and renderer.last_metrics:
            print(renderer.last_metrics.format_summary())
        else:
            print("No metrics recorded yet.")
    elif name == "/trace":
        if renderer and hasattr(renderer, "tracer") and renderer.tracer:
            print(renderer.tracer.format_trace())
        else:
            print("No trace recorded yet.")
    elif name == "/think":
        try:
            if arg in ("on", "off"):
                session.set_thinking(arg == "on")
            elif arg:
                print("Usage: /think [on|off]")
                return True
            print(f"Thinking: {'on' if session.spec.think else 'off'}")
        except (ValueError, OllamaError) as e:
            print(f"[error] {e}", file=sys.stderr)
    elif name == "/model":
        try:
            if arg:
                was_thinking = session.spec.think
                session.set_model(arg)
                print(f"Model: {arg} (history kept)")
                if was_thinking and not session.spec.think:
                    print(f"Thinking turned off: {arg} can't reason before answering.")
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