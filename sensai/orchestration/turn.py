"""run_turn(): one learner turn as a stream of events (layer 2).

Order: TurnStarted -> [GuardrailTriggered] -> AgentRunStarted -> [Thinking...] -> Token...
-> AgentRunFinished -> TurnFinished. Every turn ends with exactly one TurnFinished, even when it fails, so surfaces
always know when a turn is over. One agent for now; the orchestrator (layer 4) will run several
agents inside the same turn, each with its own AgentRunStarted/Finished pair.
"""
import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator

from ..core.agent_spec import AgentSpec
from ..core.events import (AgentRunFinished, AgentRunStarted, Error, Event, GuardrailTriggered,
                           Thinking, Token, TurnFinished, TurnStarted)
from ..core.ids import new_id
from ..core.messages import Message
from ..llm.ollama import ChatStats, OllamaError, Thought
from ..trust import guardrails
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
    run_id = new_id()
    yield AgentRunStarted(turn_id=turn_id, run_id=run_id, agent=spec.name)

    reply, ttft_ms, ok, stats = "", None, True, None
    try:
        async for token in stream_tokens(session.client, spec, build_context(session)):
            if isinstance(token, ChatStats):
                stats = token
                continue
            if ttft_ms is None:                     # first output of any kind, reasoning included
                ttft_ms = (time.perf_counter_ns() - started) / 1e6
            if isinstance(token, Thought):
                yield Thinking(turn_id=turn_id, run_id=run_id, text=token.text)
                continue
            reply += token
            yield Token(turn_id=turn_id, run_id=run_id, text=token)
    except OllamaError as e:
        ok = False
        session.history.pop()           # drop the unanswered message
        yield Error(turn_id=turn_id, kind=e.kind, message=str(e))
    except (asyncio.CancelledError, KeyboardInterrupt):
        # Stopped by the learner: keep the question and what was already written, so the model
        # knows what it had said ("continue" works). Nothing written yet: drop the question.
        if reply:
            session.history.append({"role": "assistant", "content": reply})
        else:
            session.history.pop()
        raise
    except BaseException:               # unexpected errors, abandoned stream
        session.history.pop()
        raise

    if ok:
        session.history.append({"role": "assistant", "content": reply})
    yield AgentRunFinished(turn_id=turn_id, run_id=run_id, agent=spec.name, ok=ok)
    yield _turn_finished(turn_id, spec.model, started, ttft_ms, stats)


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
    """System prompt + the last turns. Stub: replaced by the context builder in layer 4."""
    max_turns = session.config["memory"]["max_history_turns"]
    return [{"role": "system", "content": session.spec.system_prompt}] + session.history[-max_turns * 2:]


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


class _Done:
    def __init__(self, stats: ChatStats | None):
        self.stats = stats


async def stream_tokens(client, spec: AgentSpec, messages: list[Message]
                        ) -> AsyncIterator[str | Thought | ChatStats]:
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
        tokens: Iterator[str] = client.chat_stream(spec.model, messages, num_ctx=spec.num_ctx,
                                                   temperature=spec.temperature, think=spec.think)
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
