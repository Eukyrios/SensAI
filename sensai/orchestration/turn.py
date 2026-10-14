# run_turn(): one learner turn as a stream of events, and the thread bridge to the transport.

import asyncio
import threading
from contextlib import aclosing
import time
from collections.abc import AsyncIterator, Callable, Iterator
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


class InvalidStructuredOutput(Exception):
    kind = "invalid_output"


async def run_turn(session: Session, text: str) -> AsyncIterator[Event]:
    turn = _Turn(session)
    yield TurnStarted(turn_id=turn.turn_id, session_id=session.id, user_input=text)
    learner_text = text.strip()
    if not learner_text:
        yield Error(turn_id=turn.turn_id, kind="empty_input", message="Empty message")
        yield turn.finished_event()
        return
    clean_text = guardrails.filter_input(learner_text, session.config)
    if clean_text != learner_text:
        yield turn.guardrail_redacted_event()
    turn.save_question(clean_text)
    yield turn.run_started_event()
    async with aclosing(_answer_safely(turn)) as answer_events:
        async for event in answer_events:
            yield event
    yield turn.run_finished_event()
    yield turn.finished_event()


async def _answer_safely(turn: "_Turn") -> AsyncIterator[Event]:
    try:
        async for event in turn.answer():
            yield event
    except (OllamaError, InvalidStructuredOutput) as error:
        turn.drop_question()
        yield Error(turn_id=turn.turn_id, kind=error.kind, message=str(error))
        return
    except (asyncio.CancelledError, KeyboardInterrupt):
        turn.keep_partial_answer()
        raise
    except BaseException:
        turn.drop_question()
        raise
    turn.save_answer()


class _Turn:

    def __init__(self, session: Session):
        self.session = session
        self.spec: AgentSpec = session.spec
        self.turn_id = new_id()
        self.run_id = new_id()
        self.started_ns = time.perf_counter_ns()
        self.answer_text = ""
        self.first_output_ms: float | None = None
        self.stats: ChatStats | None = None
        self.succeeded = False
        self.question_id: str | None = None
        self.question_parent_id: str | None = None

    def save_question(self, text: str) -> None:
        self.session.history.append({"role": "user", "content": text})
        self.question_parent_id = self.session.leaf
        self.question_id = self.session.record("user", text)

    def drop_question(self) -> None:
        self.session.history.pop()
        self.session.discard(self.question_id, self.question_parent_id)

    def save_answer(self) -> None:
        self.session.history.append({"role": "assistant", "content": self.answer_text})
        self.session.record("assistant", self.answer_text)
        self.succeeded = True

    def keep_partial_answer(self) -> None:
        if self.answer_text:
            self.save_answer()
        else:
            self.drop_question()

    async def answer(self) -> AsyncIterator[Event]:
        messages = build_context(self.session)
        async for event in self._call_model(messages):
            yield event
        if self.spec.output_schema is not None:
            async for event in self._validated_output(messages):
                yield event

    async def _call_model(self, messages: list[Message]) -> AsyncIterator[Event]:
        self.answer_text = ""
        request_format = structured.request_format(self.spec)
        async for piece in stream_tokens(self.session.client, self.spec, messages, request_format):
            if isinstance(piece, ChatStats):
                self.stats = _sum_stats(self.stats, piece)
                continue
            self._note_first_output()
            event = self._event_for(piece)
            if event is not None:
                yield event

    def _event_for(self, piece: str | Thought) -> Event | None:
        if isinstance(piece, Thought):
            return Thinking(turn_id=self.turn_id, run_id=self.run_id, text=piece.text)
        self.answer_text += piece
        if self.spec.output_schema is not None:
            return None
        return Token(turn_id=self.turn_id, run_id=self.run_id, text=piece)

    def _note_first_output(self) -> None:
        if self.first_output_ms is None:
            self.first_output_ms = (time.perf_counter_ns() - self.started_ns) / 1e6

    async def _validated_output(self, messages: list[Message]) -> AsyncIterator[Event]:
        schema = self.spec.output_schema
        data, problems = structured.validate_reply(self.answer_text, schema)
        attempts = 1
        if problems:
            attempts = 2
            async for event in self._call_model(_repair_messages(messages, self.answer_text, problems)):
                yield event
            data, problems = structured.validate_reply(self.answer_text, schema)
        if problems:
            raise InvalidStructuredOutput(structured.failure_message(problems, self.answer_text))
        yield StructuredOutput(turn_id=self.turn_id, run_id=self.run_id, data=data, attempts=attempts)

    def guardrail_redacted_event(self) -> GuardrailTriggered:
        return GuardrailTriggered(turn_id=self.turn_id, side="input", rule="filter_input", action="redact")

    def run_started_event(self) -> AgentRunStarted:
        return AgentRunStarted(turn_id=self.turn_id, run_id=self.run_id, agent=self.spec.name)

    def run_finished_event(self) -> AgentRunFinished:
        return AgentRunFinished(turn_id=self.turn_id, run_id=self.run_id, agent=self.spec.name,
                                ok=self.succeeded)

    def finished_event(self) -> TurnFinished:
        if self.stats is None:
            return TurnFinished(turn_id=self.turn_id, model=self.spec.model, ttft_ms=self.first_output_ms,
                                total_duration_ns=time.perf_counter_ns() - self.started_ns)
        return TurnFinished(turn_id=self.turn_id, model=self.spec.model, ttft_ms=self.first_output_ms,
                            prompt_eval_count=self.stats.prompt_eval_count,
                            eval_count=self.stats.eval_count,
                            total_duration_ns=self.stats.total_duration_ns)


def _repair_messages(messages: list[Message], invalid_reply: str, problems: list[str]) -> list[Message]:
    return messages + [{"role": "assistant", "content": invalid_reply},
                       {"role": "user", "content": structured.repair_request(problems)}]


def _sum_stats(total: ChatStats | None, more: ChatStats) -> ChatStats:
    if total is None:
        return more
    summed = {field.name: getattr(total, field.name) + getattr(more, field.name)
              for field in fields(ChatStats) if field.name != "done_reason"}
    return ChatStats(**summed, done_reason=more.done_reason)


def build_context(session: Session) -> list[Message]:
    max_turns = session.config["memory"]["max_history_turns"]
    profile = [session.profile] if session.profile else []
    return ([{"role": "system", "content": _system_prompt(session.spec)}]
            + profile
            + session.history[-max_turns * 2:])


def _system_prompt(spec: AgentSpec) -> str:
    if spec.output_schema is None:
        return spec.system_prompt
    return f"{spec.system_prompt}\n\n{structured.schema_instructions(spec.output_schema)}".strip()


async def stream_tokens(client, spec: AgentSpec, messages: list[Message],
                        format: dict | str | None = None) -> AsyncIterator[str | Thought | ChatStats]:
    options = {"format": format} if format is not None else {}

    def open_stream() -> Iterator[str | Thought]:
        return client.chat_stream(spec.model, messages, num_ctx=spec.num_ctx,
                                  temperature=spec.temperature, think=spec.think, **options)

    stream = _ThreadedStream(asyncio.get_running_loop())
    stream.start(open_stream)
    try:
        async for item in stream.items():
            yield item
    finally:
        stream.stop()


class _StreamFailed:
    def __init__(self, error: BaseException):
        self.error = error


class _StreamEnded:
    def __init__(self, stats: ChatStats | None):
        self.stats = stats


class _ThreadedStream:

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop_requested = threading.Event()

    def start(self, open_stream: Callable[[], Iterator[str | Thought]]) -> None:
        threading.Thread(target=self._produce, args=(open_stream,), name="ollama-stream",
                         daemon=True).start()

    def stop(self) -> None:
        self.stop_requested.set()

    async def items(self) -> AsyncIterator[str | Thought | ChatStats]:
        while True:
            item = await self.queue.get()
            if isinstance(item, _StreamEnded):
                if item.stats is not None:
                    yield item.stats
                return
            if isinstance(item, _StreamFailed):
                raise item.error
            yield item

    def _produce(self, open_stream: Callable[[], Iterator[str | Thought]]) -> None:
        source = open_stream()
        try:
            self._forward(source)
        except BaseException as error:
            self._hand_over(_StreamFailed(error))
        finally:
            source.close()

    def _forward(self, source: Iterator[str | Thought]) -> None:
        while not self.stop_requested.is_set():
            try:
                piece = next(source)
            except StopIteration as end:
                self._hand_over(_StreamEnded(end.value))
                return
            self._hand_over(piece)

    def _hand_over(self, item) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:
            pass
