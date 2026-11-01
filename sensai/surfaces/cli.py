# Command-line chat: live input line, slash commands and the CLI renderer.

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from ..core.agent_spec import SpecError
from ..core.persona import persona_for, persona_names
from ..core.config import load_config, resolve
from ..core.errors import ErrorKind
from ..core.events import (AgentRunStarted, ContextCompressed, Error, Event, FaithfulnessChecked, GuardrailTriggered, HandoffEvent,
                           StructuredOutput, Thinking, Token, ToolCall, ToolResult, TurnFinished)
from ..core.handoff import Handoff
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
    "/persona": ("/persona [name]", "show the personas, or switch persona (history kept)"),
    "/think": ("/think [on|off]", "show whether the model reasons before answering, or switch it"),
    "/metrics": ("/metrics", "show token usage, speed, and cost estimation for the last turn"),
    "/trace": ("/trace", "show the agent runs of the last turn"),
    "/reset": ("/reset", "clear the history and start a new session (the old one stays stored)"),
    "/quit": ("/quit", "exit (Ctrl-D works too)"),
}
HELP = "Commands: " + "  ".join(usage for usage, _ in COMMANDS.values())


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    with _exit_on_startup_error(args):
        config = load_config(args.config, cli={"agents": {"model_override": args.model, "default": args.agent},
                                               "router": {"enabled": False if args.agent else None}})
        if args.list_sessions:
            list_sessions(config)
            return
        session = _open_session(config, args)
    try:
        _print_banner(session, args)
        _start_knowledge(session)
        _chat(session, JsonlLogger(resolve(config["logging"]["path"])))
    finally:
        session.knowledge.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai, a local Japanese tutor")
    parser.add_argument("--model", help="Ollama model for every agent (overrides the agent specs)")
    parser.add_argument("--agent", help="agent to talk to, from agents/<name>.toml (default: sensei)")
    parser.add_argument("--config", help="path to a config.toml")
    parser.add_argument("--session", metavar="ID", help="resume a stored session (see --list-sessions)")
    parser.add_argument("--list-sessions", action="store_true", help="list the stored sessions and exit")
    return parser.parse_args(argv)


@contextmanager
def _exit_on_startup_error(args: argparse.Namespace) -> Iterator[None]:
    try:
        yield
    except FileNotFoundError as error:
        sys.exit(f"[error] config not found: {error.filename}")
    except SessionNotFound:
        sys.exit(f"[error] session not found: {args.session} (see --list-sessions)")
    except (SpecError, OllamaError) as error:
        sys.exit(f"[error] {error}")


def _open_session(config: dict, args: argparse.Namespace) -> Session:
    session = Session.resume(config, args.session) if args.session else Session(config)
    session.client.require_model(session.spec.model)
    return session


def _print_banner(session: Session, args: argparse.Namespace) -> None:
    print(f"Sensai ({session.spec.model}) — type /help for commands")
    if args.session:
        print(f"Resumed session {session.id} ({len(session.history)} messages)")


def _start_knowledge(session: Session) -> None:
    servers = [settings["name"] for settings in session.knowledge.enabled_servers()]
    if servers:
        print(dim(f"[mcp] starting {', '.join(servers)}…"), flush=True)
    session.knowledge.start()
    for name, tools in session.knowledge.server_tools.items():
        print(dim(f"[mcp] {name}: {', '.join(tools)}"))
    for name, reason in session.knowledge.unavailable.items():
        print(f"[mcp] {name} unavailable, continuing without it: {reason}", file=sys.stderr)


def _chat(session: Session, logger: JsonlLogger) -> None:
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
    renderer = CLIRenderer(PlainOutput())
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if not asyncio.run(handle_line_and_settle(session, line, renderer, logger)):
                return
        except KeyboardInterrupt:
            continue


async def handle_line_and_settle(session: Session, line: str, renderer: "CLIRenderer",
                                 logger: JsonlLogger | None = None) -> bool:
    keep_going = await handle_line(session, line, renderer, logger)
    await session.settle()
    return keep_going


async def live_chat(session: Session, logger: JsonlLogger | None = None) -> None:
    await LiveChat(session, logger).run()


class LiveChat:

    def __init__(self, session: Session, logger: JsonlLogger | None):
        self.session = session
        self.logger = logger
        self.pending_lines: asyncio.Queue[str | None] = asyncio.Queue()
        self.status: str | None = None
        self.current_turn: asyncio.Task | None = None
        self.quit_requested = False
        self.out = LiveOutput()
        self.out.on_change = self.redraw
        self.renderer = CLIRenderer(self.out, self.set_status)
        self.prompt = PromptSession(erase_when_done=True, bottom_toolbar=self.toolbar,
                                    key_bindings=self._key_bindings())
        self.prompt.app.ttimeoutlen = 0.05

    async def run(self) -> None:
        with patch_stdout(raw=True):
            reader = asyncio.create_task(self.read_lines())
            try:
                await self.process_lines()
            finally:
                reader.cancel()
                await self.session.settle()

    def toolbar(self) -> str:
        if self.status:
            parts = [f"{self.status}  ·  Esc: stop the reply"]
        else:
            parts = ["Enter: send  ·  Esc: stop a reply  ·  Ctrl-D: quit  ·  /help"]
        if self.pending_lines.qsize():
            parts.append(f"{self.pending_lines.qsize()} queued")
        return "  ·  ".join(parts)

    def prompt_message(self) -> ANSI:
        return ANSI(f"{self.out.partial}\n> " if self.out.partial else "> ")

    def is_replying(self) -> bool:
        return self.current_turn is not None and not self.current_turn.done()

    def stop_reply(self) -> None:
        if self.is_replying():
            self.current_turn.cancel()

    def redraw(self) -> None:
        if self.prompt.app.is_running:
            self.prompt.app.invalidate()

    def set_status(self, status: str | None) -> None:
        self.status = status
        self.redraw()

    def _key_bindings(self) -> KeyBindings:
        keys = KeyBindings()
        keys.add("escape", filter=Condition(self.is_replying), eager=True)(self._on_escape)
        keys.add("c-c")(self._ignore_key)
        return keys

    def _on_escape(self, event) -> None:
        self.stop_reply()

    def _ignore_key(self, event) -> None:
        return None

    async def read_lines(self) -> None:
        while True:
            try:
                line = await self.prompt.prompt_async(self.prompt_message)
            except EOFError:
                self._quit_now()
                return
            await self.pending_lines.put(line)

    def _quit_now(self) -> None:
        self.quit_requested = True
        self.stop_reply()
        self.pending_lines.put_nowait(None)

    async def process_lines(self) -> None:
        while not self.quit_requested and (line := await self.pending_lines.get()) is not None:
            if line.strip() and not await self.handle(line.strip()):
                return

    async def handle(self, line: str) -> bool:
        print(f"\n> {line}")
        self.current_turn = asyncio.create_task(handle_line(self.session, line, self.renderer, self.logger))
        try:
            return await self.current_turn
        except asyncio.CancelledError:
            return True
        finally:
            self.current_turn = None
            self.set_status(None)


class PlainOutput:

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

    def __init__(self) -> None:
        self.partial = ""
        self.on_change: Callable[[], None] = _do_nothing

    def write(self, text: str) -> None:
        *finished_lines, self.partial = (self.partial + text).split("\n")
        for line in finished_lines:
            print(line)
        self.on_change()

    def end_line(self) -> None:
        if self.partial:
            print(self.partial)
            self.partial = ""
            self.on_change()


Output = PlainOutput | LiveOutput


def _do_nothing() -> None:
    return None


def _ignore_status(status: str | None) -> None:
    return None


async def handle_line(session: Session, line: str, renderer: "CLIRenderer",
                      logger: JsonlLogger | None = None) -> bool:
    line = line.strip()
    if not line:
        return True
    if line.startswith("/"):
        return run_command(session, line, renderer)
    await render_turn(session, line, renderer, logger)
    return True


async def render_turn(session: Session, text: str, renderer: "CLIRenderer",
                      logger: JsonlLogger | None = None) -> None:
    stream = run_turn(session, text)
    if logger is not None:
        stream = logger.log_stream(stream)
    await renderer.render_stream(stream)


class CLIRenderer:

    def __init__(self, out: Output | None = None, status: Status | None = None) -> None:
        self.out = out or PlainOutput()
        self.status = status or _ignore_status
        self.tracer = TurnTracer()
        self.last_metrics = None
        self.showing_reasoning = False
        self.answer_started = False

    async def render_stream(self, stream: AsyncIterator[Event]) -> None:
        self.showing_reasoning = self.answer_started = False
        try:
            async for event in stream:
                self.tracer.process_event(event)
                self.render(event)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.out.end_line()
            print("[interrupted]", file=sys.stderr)
            raise

    def render(self, event: Event) -> None:
        match event:
            case AgentRunStarted():
                self.status("thinking…")
            case Thinking(text=text):
                self._show_reasoning(text)
            case Token(text=text):
                self._show_answer_text(text)
            case ContextCompressed() as compressed:
                self.out.end_line()
                print(dim(compression_line(compressed)))
            case GuardrailTriggered(side=side, rule=rule, action=action):
                self.out.end_line()
                print(f"[guardrail] {side} {action}: {rule}", file=sys.stderr)
            case Error(kind=kind, message=message):
                self.out.end_line()
                self._render_error(kind, message)
            case StructuredOutput(data=data, attempts=attempts):
                self._show_structured_output(data, attempts)
            case HandoffEvent(handoff=handoff):
                self._show_handoff(handoff)
            case ToolCall(name=name, args=args):
                self.out.end_line()
                print(dim(f"[tool] {name} {json.dumps(args, ensure_ascii=False)}"))
            case ToolResult(name=name, ok=False, result=result):
                print(dim(f"[tool] {name} failed: {result[:200]}"))
            case TurnFinished() as finished:
                self._finish_turn(finished)
            case FaithfulnessChecked() as check:
                self._show_faithfulness(check)

    def _show_faithfulness(self, check: FaithfulnessChecked) -> None:
        self.out.end_line()
        if check.error:
            print(dim(f"[faithfulness] not checked: {check.error}"), file=sys.stderr)
            return
        relevance = f" · relevance {check.relevance:.2f}" if check.relevance is not None else ""
        print(dim(f"[faithfulness] {check.faithfulness:.2f}{relevance} · {check.supported} supported, "
                  f"{check.contradicted} contradicted, {check.unverified} unverified (judge {check.judge_model})"))
        for claim in check.flagged:
            print(f"  [{claim['status'].lower()}] {claim['claim']}" + (f" — {claim['reasoning']}" if claim["reasoning"] else ""))

    def _show_reasoning(self, text: str) -> None:
        if not self.showing_reasoning:
            self.out.write(dim("thinking:") + "\n")
            self.showing_reasoning = True
        self.out.write(dim_lines(text))

    def _show_answer_text(self, text: str) -> None:
        self.status("replying…")
        if self.showing_reasoning and not self.answer_started:
            self.out.end_line()
            self.out.write("\n")
        self.answer_started = True
        self.out.write(text)

    def _show_structured_output(self, data: dict, attempts: int) -> None:
        self.out.end_line()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        if attempts > 1:
            print(dim("(repaired: the first reply didn't match the schema)"))

    def _show_handoff(self, handoff: Handoff) -> None:
        self.out.end_line()
        print(dim(f"[handoff] {handoff.from_agent} → {handoff.to_agent}: {handoff.reason}"))
        self.answer_started = self.showing_reasoning = False

    def _finish_turn(self, finished: TurnFinished) -> None:
        self.status(None)
        self.out.end_line()
        if finished.eval_count:
            self.last_metrics = compute_turn_metrics(finished)
            print(dim(usage_line(finished)))

    def _render_error(self, kind: ErrorKind | str, message: str) -> None:
        if kind == "model_missing":
            print(f"\n[error] Model not found. Run:\n  {message}", file=sys.stderr)
        elif kind == "connection":
            print("\n[error] Cannot reach Ollama. Start it with:\n  ollama serve", file=sys.stderr)
        elif kind == "timeout":
            print(f"\n[error] Request timed out: {message}", file=sys.stderr)
        else:
            print(f"\n[error] {message}", file=sys.stderr)


def usage_line(finished: TurnFinished) -> str:
    parts = [f"{finished.prompt_eval_count} prompt + {finished.eval_count} reply tokens"]
    if finished.ttft_ms is not None:
        parts.append(f"first token {finished.ttft_ms / 1000:.1f}s")
    parts.append(f"{finished.total_duration_ns / 1e9:.1f}s")
    return " · ".join(parts)


def compression_line(event: ContextCompressed) -> str:
    return (f"[context] compressed {event.messages} messages: "
            f"{event.tokens_before:,} -> {event.tokens_after:,} tokens")


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if sys.stdout.isatty() else text


def dim_lines(text: str) -> str:
    return "\n".join(dim(part) if part else part for part in text.split("\n"))


def run_command(session: Session, line: str, renderer: CLIRenderer | None = None) -> bool:
    name, _, argument = line.partition(" ")
    if name == "/quit":
        return False
    handler = COMMAND_HANDLERS.get(name)
    if handler is None:
        print(f"Unknown command {name}. Type /help for the list.")
    else:
        handler(session, argument.strip(), renderer)
    return True


def _show_help(session: Session, argument: str, renderer: CLIRenderer | None) -> None:
    width = max(len(usage_text) for usage_text, _ in COMMANDS.values())
    print("\n".join(f"  {usage_text:<{width}}  {description}" for usage_text, description in COMMANDS.values()))


def _reset_history(session: Session, argument: str, renderer: CLIRenderer | None) -> None:
    session.reset()
    print("History cleared.")


def _show_metrics(session: Session, argument: str, renderer: CLIRenderer | None) -> None:
    if renderer is not None and renderer.last_metrics:
        print(renderer.last_metrics.format_summary())
    else:
        print("No metrics recorded yet.")


def _show_trace(session: Session, argument: str, renderer: CLIRenderer | None) -> None:
    if renderer is not None and renderer.tracer:
        print(renderer.tracer.format_trace())
    else:
        print("No trace recorded yet.")


def _show_or_switch_thinking(session: Session, argument: str, renderer: CLIRenderer | None) -> None:
    if argument and argument not in ("on", "off"):
        print("Usage: /think [on|off]")
        return
    try:
        if argument:
            session.set_thinking(argument == "on")
        print(f"Thinking: {'on' if session.spec.think else 'off'}")
    except (ValueError, OllamaError) as error:
        print(f"[error] {error}", file=sys.stderr)


def _show_or_switch_model(session: Session, argument: str, renderer: CLIRenderer | None) -> None:
    try:
        if argument:
            _switch_model(session, argument)
        else:
            _show_models(session)
    except OllamaError as error:
        print(f"[error] {error}", file=sys.stderr)


def _switch_model(session: Session, model: str) -> None:
    was_thinking = session.spec.think
    session.set_model(model)
    print(f"Model: {model} (history kept)")
    if was_thinking and not session.spec.think:
        print(f"Thinking turned off: {model} can't reason before answering.")


def _show_or_switch_persona(session: Session, argument: str, renderer: CLIRenderer | None) -> None:
    try:
        if argument:
            session.set_persona(argument)
            print(f"Persona: {session.persona.name} (history kept)")
        else:
            _show_personas(session)
    except SpecError as error:
        print(f"[error] {error}", file=sys.stderr)


def _show_personas(session: Session) -> None:
    for name in persona_names(session.config):
        persona = persona_for(name, session.config)
        marker = "*" if name == session.persona.name else " "
        print(f" {marker} {name:<10} {persona.description}")


def _show_models(session: Session) -> None:
    installed = ", ".join(session.client.list_models()) or "none"
    print(f"Model: {session.spec.model}\nInstalled: {installed}")


COMMAND_HANDLERS = {
    "/help": _show_help,
    "/persona": _show_or_switch_persona,
    "/reset": _reset_history,
    "/metrics": _show_metrics,
    "/trace": _show_trace,
    "/think": _show_or_switch_thinking,
    "/model": _show_or_switch_model,
}


if __name__ == "__main__":
    main()
