"""run_turn(): one learner turn as a stream of events (layer 2).

Order: TurnStarted -> [GuardrailTriggered] -> AgentRunStarted -> Token... -> AgentRunFinished
-> TurnFinished. Every turn ends with exactly one TurnFinished, even when it fails, so surfaces
always know when a turn is over. One agent for now; the orchestrator (layer 4) will run several
agents inside the same turn, each with its own AgentRunStarted/Finished pair.
"""
import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator

from ..core.agent_spec import AgentSpec
from ..core.events import (AgentRunFinished, AgentRunStarted, Error, Event, GuardrailTriggered,
                           Token, TurnFinished, TurnStarted)
from ..core.ids import new_id
from ..core.messages import Message
from ..llm.ollama import OllamaError
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

    reply, ttft_ms, ok = "", None, True
    try:
        async for token in stream_tokens(session.client, spec, build_context(session)):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter_ns() - started) / 1e6
            reply += token
            yield Token(turn_id=turn_id, run_id=run_id, text=token)
    except OllamaError as e:
        ok = False
        session.history.pop()           # drop the unanswered message
        yield Error(turn_id=turn_id, kind=e.kind, message=str(e))
    except BaseException:               # Ctrl-C, cancellation, unexpected errors
        session.history.pop()
        raise

    if ok:
        session.history.append({"role": "assistant", "content": reply})
    yield AgentRunFinished(turn_id=turn_id, run_id=run_id, agent=spec.name, ok=ok)
    # prompt_eval_count / eval_count come from Ollama's final chunk once the transport exposes them (#11).
    yield TurnFinished(turn_id=turn_id, model=spec.model, ttft_ms=ttft_ms,
                       total_duration_ns=time.perf_counter_ns() - started)


def build_context(session: Session) -> list[Message]:
    """System prompt + the last turns. Stub: replaced by the context builder in layer 4."""
    max_turns = session.config["memory"]["max_history_turns"]
    return [{"role": "system", "content": session.spec.system_prompt}] + session.history[-max_turns * 2:]


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


_DONE = object()


async def stream_tokens(client, spec: AgentSpec, messages: list[Message]) -> AsyncIterator[str]:
    """Bridge the blocking transport to async: a daemon thread reads the stream and hands tokens over.

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
            for token in tokens:
                if stop.is_set():
                    return
                hand_over(token)
        except BaseException as e:
            hand_over(_Failure(e))
            return
        finally:
            tokens.close()
        hand_over(_DONE)

    threading.Thread(target=produce, name="ollama-stream", daemon=True).start()
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stop.set()
