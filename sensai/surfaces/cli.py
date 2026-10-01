"""Command-line chat interface and CLI renderer. Run with: python -m sensai"""
import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from ..core.agent_spec import SpecError
from ..core.config import load_config
from ..core.events import (AgentRunStarted, Error, Event, GuardrailTriggered, Thinking, Token,
                           TurnFinished)
from ..llm.ollama import OllamaError
from ..orchestration.session import Session
from ..orchestration.turn import run_turn
from ..trust.logging import JsonlLogger

COMMANDS = {
    "/help": ("/help", "list the commands"),
    "/model": ("/model [name]", "show the current and installed models, or switch model (history kept)"),
    "/think": ("/think [on|off]", "show whether the model reasons before answering, or switch it"),
    "/reset": ("/reset", "clear the conversation history"),
    "/quit": ("/quit", "exit (Ctrl-D works too)"),
}


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

    logger = JsonlLogger("logs/events.jsonl")
    print(f"Sensai ({session.spec.model}) — type /help for commands")
    if sys.stdin.isatty() and sys.stdout.isatty():
        asyncio.run(live_chat(session, logger))
    else:
        plain_chat(session, logger)


Status = Callable[[str | None], None]


def plain_chat(session: Session, logger: JsonlLogger | None = None) -> None:
    """Line by line, for pipes and scripts: read a message, answer it, repeat."""
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if not asyncio.run(handle_line(session, line, PlainOutput(), logger=logger)):
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
        # The line the model is still writing is drawn here, above the input line, and redrawn in
        # place; finished lines are printed above normally. Printing half a line above a live
        # prompt would collide with it.
        return ANSI(f"{out.partial}\n> " if out.partial else "> ")

    replying = Condition(lambda: state["turn"] is not None and not state["turn"].done())
    keys = KeyBindings()

    @keys.add("escape", filter=replying, eager=True)     # only while a reply runs: normal typing keeps Esc
    def _stop_reply(event) -> None:
        state["turn"].cancel()

    @keys.add("c-c")                    # Ctrl-C does nothing: Esc stops a reply, Ctrl-D quits
    def _ignore(event) -> None:
        pass

    prompt = PromptSession(erase_when_done=True, bottom_toolbar=toolbar, key_bindings=keys)
    prompt.app.ttimeoutlen = 0.05       # a lone Esc is recognised at once, not after half a second

    def redraw() -> None:
        if prompt.app.is_running:
            prompt.app.invalidate()

    def set_status(status: str | None) -> None:
        state["status"] = status
        redraw()

    out.on_change = redraw

    async def read_lines() -> None:
        while True:
            try:
                line = await prompt.prompt_async(message)
            except EOFError:                # Ctrl-D quits now: stop any reply, skip queued questions
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
                print(f"\n> {line.strip()}")       # shown when it actually runs
                state["turn"] = asyncio.create_task(handle_line(session, line, out, set_status, logger))
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


async def handle_line(session: Session, line: str, out: Output, status: Status | None = None,
                      logger: JsonlLogger | None = None) -> bool:
    """One input line: a /command or a message for the model. Returns False when the chat should stop."""
    line = line.strip()
    if not line:
        return True
    if line.startswith("/"):
        return run_command(session, line)
    await render_turn(session, line, out, status, logger)
    return True


async def render_turn(session: Session, text: str, out: Output, status: Status | None = None,
                      logger: JsonlLogger | None = None) -> None:
    """One turn: run_turn -> (logger, which writes each event and passes it on) -> CLI renderer."""
    stream = run_turn(session, text)
    if logger is not None:
        stream = logger.log_stream(stream)
    await CLIRenderer(out, status).render_stream(stream)


class CLIRenderer:
    """Consumes events from an Event stream and renders them to the CLI.

    `status` receives "thinking…" until the first token, "replying…" while text streams, then None.
    The model's reasoning, when there is some, is shown dim before the answer. After each reply,
    a dim line shows the token usage.
    """

    def __init__(self, out: Output | None = None, status: Status | None = None) -> None:
        self.out = out or PlainOutput()
        self.status = status or (lambda _: None)

    async def render_stream(self, stream: AsyncIterator[Event]) -> None:
        out, status = self.out, self.status
        reasoning, answering = False, False
        async for event in stream:
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
                        out.write("\n")             # a blank line between reasoning and answer
                    answering = True
                    out.write(t)
                case GuardrailTriggered(side=side, rule=rule, action=action):
                    out.end_line()
                    print(f"[guardrail] {side} {action}: {rule}", file=sys.stderr)
                case Error(message=m):
                    out.end_line()
                    print(f"[error] {m}", file=sys.stderr)
                case TurnFinished() as finished:
                    status(None)
                    out.end_line()
                    if finished.eval_count:
                        print(dim(usage(finished)))


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
    """Dim each line on its own, so a line printed alone never leaves the terminal dimmed."""
    return "\n".join(dim(part) if part else part for part in text.split("\n"))


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
