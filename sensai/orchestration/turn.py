"""run_turn(): one learner turn as a stream of events (layer 2).

Order: TurnStarted -> [GuardrailTriggered] -> AgentRunStarted -> [Thinking...] -> Token...
-> AgentRunFinished -> TurnFinished. An agent with an output_schema emits no Token (its reply is
data, not text for the learner) but one StructuredOutput, or Error(kind="invalid_output"). Every turn ends with exactly one TurnFinished, even when it fails, so surfaces
always know when a turn is over. One agent for now; the orchestrator (layer 4) will run several
agents inside the same turn, each with its own AgentRunStarted/Finished pair.
"""
import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import fields

from ..core.agent_spec import AgentSpec
from ..core.events import (AgentRunFinished, AgentRunStarted, Error, Event, GuardrailTriggered,
                           StructuredOutput, Thinking, Token, TurnFinished, TurnStarted)
from ..core.ids import new_id
from ..core.messages import Message
from ..llm.ollama import ChatStats, OllamaError, Thought
from ..trust import guardrails
from . import structured
from .session import Session


async def run_turn(session: Session, text: str) -> AsyncIterator[Event]:
    turn_id = new_id()
    spec = session.spec
    started = time.perf_counter_ns()
    yield TurnStarted(turn_id=turn_id, session_id=session.id, user_input=text)

    text = text.strip()
    if not text:
        yield Error(turn_id=turn_id, kind="empty_input", message="Empty message")
        yield TurnFinished(turn_id=turn_id, model=spec.model,
                           total_duration_ns=time.perf_counter_ns() - started)
        return

    clean = guardrails.filter_input(text, session.config)
    if clean != text:
        yield GuardrailTriggered(turn_id=turn_id, side="input", rule="filter_input", action="redact")

    session.history.append({"role": "user", "content": clean})
    parent_id = session.leaf
    user_id = session.record("user", clean)         # saved before the model is called
    run_id = new_id()
    yield AgentRunStarted(turn_id=turn_id, run_id=run_id, agent=spec.name)

    run, ok = _Run(started), True
    messages = build_context(session)
    try:
        async for event in _call_model(session, spec, messages, run, turn_id, run_id):
            yield event
        if spec.output_schema is not None:
            async for event in _validated(session, spec, messages, run, turn_id, run_id):
                yield event
    except OllamaError as e:
        ok = False
        session.history.pop()           # drop the unanswered message
        session.discard(user_id, parent_id)
        yield Error(turn_id=turn_id, kind=e.kind, message=str(e))
    except _InvalidOutput as e:
        ok = False
        session.history.pop()
        session.discard(user_id, parent_id)
        yield Error(turn_id=turn_id, kind="invalid_output", message=str(e))
    except (asyncio.CancelledError, KeyboardInterrupt):
        # Stopped by the learner: keep the question and what was already written, so the model
        # knows what it had said ("continue" works). Nothing written yet: drop the question.
        if run.text:
            session.history.append({"role": "assistant", "content": run.text})
            session.record("assistant", run.text)
        else:
            session.history.pop()
            session.discard(user_id, parent_id)
        raise
    except BaseException:               # unexpected errors, abandoned stream
        session.history.pop()
        session.discard(user_id, parent_id)
        raise

    if ok:
        session.history.append({"role": "assistant", "content": run.text})
        session.record("assistant", run.text)
    yield AgentRunFinished(turn_id=turn_id, run_id=run_id, agent=spec.name, ok=ok)
    yield _turn_finished(turn_id, spec.model, started, run.ttft_ms, run.stats)


class _Run:
    """What one agent run has produced so far, across its model calls."""

    def __init__(self, started: int):
        self.started = started
        self.text = ""                          # the latest call's reply
        self.ttft_ms: float | None = None
        self.stats: ChatStats | None = None     # summed over the calls (a repair makes two)

    def add_stats(self, stats: ChatStats) -> None:
        if self.stats is None:
            self.stats = stats
            return
        summed = {f.name: getattr(self.stats, f.name) + getattr(stats, f.name)
                  for f in fields(ChatStats) if f.name != "done_reason"}
        self.stats = ChatStats(**summed, done_reason=stats.done_reason)


class _InvalidOutput(Exception):
    pass


async def _call_model(session: Session, spec: AgentSpec, messages: list[Message], run: _Run,
                      turn_id: str, run_id: str) -> AsyncIterator[Event]:
    """One model call. Text is streamed as Token events, except for a structured agent."""
    run.text = ""
    async for token in stream_tokens(session.client, spec, messages, structured.request_format(spec)):
        if isinstance(token, ChatStats):
            run.add_stats(token)
            continue
        if run.ttft_ms is None:                 # first output of any kind, reasoning included
            run.ttft_ms = (time.perf_counter_ns() - run.started) / 1e6
        if isinstance(token, Thought):
            yield Thinking(turn_id=turn_id, run_id=run_id, text=token.text)
            continue
        run.text += token
        if spec.output_schema is None:
            yield Token(turn_id=turn_id, run_id=run_id, text=token)


async def _validated(session: Session, spec: AgentSpec, messages: list[Message], run: _Run,
                     turn_id: str, run_id: str) -> AsyncIterator[Event]:
    """Validate a structured reply; on failure, send the errors back once. Raises _InvalidOutput."""
    data, problems = structured.check(run.text, spec.output_schema)
    attempts = 1
    if problems:
        attempts = 2
        retry = messages + [{"role": "assistant", "content": run.text},
                            {"role": "user", "content": structured.repair_request(problems)}]
        async for event in _call_model(session, spec, retry, run, turn_id, run_id):
            yield event
        data, problems = structured.check(run.text, spec.output_schema)
    if problems:
        raise _InvalidOutput(structured.failure_message(problems, run.text))
    yield StructuredOutput(turn_id=turn_id, run_id=run_id, data=data, attempts=attempts)


def _turn_finished(turn_id: str, model: str, started: int, ttft_ms: float | None,
                   stats: ChatStats | None) -> TurnFinished:
    """Token counts and duration from Ollama's final chunk; measured duration if there was none."""
    if stats is None:
        return TurnFinished(turn_id=turn_id, model=model, ttft_ms=ttft_ms,
                            total_duration_ns=time.perf_counter_ns() - started)
    return TurnFinished(turn_id=turn_id, model=model, ttft_ms=ttft_ms,
                        prompt_eval_count=stats.prompt_eval_count, eval_count=stats.eval_count,
                        total_duration_ns=stats.total_duration_ns)


def build_context(session: Session) -> list[Message]:
    """System prompt + learner profile + the last turns. Stub: replaced by the context builder in layer 4.

    An agent with an output_schema gets the schema and the JSON-only rule in its system prompt.
    """
    max_turns = session.config["memory"]["max_history_turns"]
    system = session.spec.system_prompt
    if session.spec.output_schema is not None:
        system = f"{system}\n\n{structured.instructions(session.spec.output_schema)}".strip()
    return ([{"role": "system", "content": system}]
            + ([session.profile] if session.profile else [])
            + session.history[-max_turns * 2:])


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


class _Done:
    def __init__(self, stats: ChatStats | None):
        self.stats = stats


async def stream_tokens(client, spec: AgentSpec, messages: list[Message],
                        format: dict | str | None = None) -> AsyncIterator[str | Thought | ChatStats]:
    """Bridge the blocking transport to async: a daemon thread reads the stream and hands tokens over.

    Yields what the transport yields (text tokens, and Thought items when the model reasons), then
    the ChatStats of the final chunk if the transport returned one.

    The transport (requests) blocks, so it runs in a thread and the event loop stays free. When the
    consumer stops early (Ctrl-C, cancellation), the thread stops at the next token and closes the
    stream, so Ollama isn't left generating an answer nobody reads.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def hand_over(item) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:            # the loop is already closed: nobody is listening
            pass

    def produce() -> None:
        options = {"format": format} if format is not None else {}
        tokens: Iterator[str] = client.chat_stream(spec.model, messages, num_ctx=spec.num_ctx,
                                                   temperature=spec.temperature, think=spec.think,
                                                   **options)
        try:
            while not stop.is_set():
                try:
                    token = next(tokens)
                except StopIteration as end:     # the generator's return value is the ChatStats
                    hand_over(_Done(end.value))
                    return
                hand_over(token)
        except BaseException as e:
            hand_over(_Failure(e))
        finally:
            tokens.close()

    threading.Thread(target=produce, name="ollama-stream", daemon=True).start()
    try:
        while True:
            item = await queue.get()
            if isinstance(item, _Done):
                if item.stats is not None:
                    yield item.stats
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stop.set()
