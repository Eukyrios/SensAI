# Agent executor: runs one AgentSpec on a list of messages and emits that run's events, from
# AgentRunStarted to AgentRunFinished. A model or output failure becomes an Error event and leaves
# the run not ok; cancellation propagates. What happens to the learner's history is the caller's call.

import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import fields

from ..core.agent_spec import AgentSpec
from ..core.errors import ErrorKind
from ..core.events import (AgentRunFinished, AgentRunStarted, Error, Event, StructuredOutput, Thinking,
                           Token)
from ..core.ids import new_id
from ..core.messages import Message
from ..llm.ollama import ChatStats, OllamaError, Thought
from . import structured
from .stream import stream_tokens


class InvalidStructuredOutput(Exception):
    kind = "invalid_output"


class AgentRun:

    def __init__(self, client, spec: AgentSpec, turn_id: str, parent_run_id: str | None = None):
        self.client = client
        self.spec = spec
        self.turn_id = turn_id
        self.run_id = new_id()
        self.parent_run_id = parent_run_id
        self.text = ""
        self.data: dict | None = None
        self.stats: ChatStats | None = None
        self.first_output_ns: int | None = None
        self.ok = False
        self.error_kind: ErrorKind | None = None

    async def events(self, messages: list[Message]) -> AsyncIterator[Event]:
        yield AgentRunStarted(turn_id=self.turn_id, run_id=self.run_id, agent=self.spec.name,
                              parent_run_id=self.parent_run_id)
        async with aclosing(self._answer_safely(messages)) as answer_events:
            async for event in answer_events:
                yield event
        yield AgentRunFinished(turn_id=self.turn_id, run_id=self.run_id, agent=self.spec.name, ok=self.ok)

    async def _answer_safely(self, messages: list[Message]) -> AsyncIterator[Event]:
        try:
            async for event in self._answer(messages):
                yield event
        except (OllamaError, InvalidStructuredOutput) as error:
            self.error_kind = error.kind
            yield Error(turn_id=self.turn_id, kind=error.kind, message=str(error))
            return
        self.ok = True

    async def _answer(self, messages: list[Message]) -> AsyncIterator[Event]:
        async for event in self._call_model(messages):
            yield event
        if self.spec.output_schema is not None:
            async for event in self._validated_output(messages):
                yield event

    async def _call_model(self, messages: list[Message]) -> AsyncIterator[Event]:
        self.text = ""
        request_format = structured.request_format(self.spec)
        async for piece in stream_tokens(self.client, self.spec, messages, request_format):
            if isinstance(piece, ChatStats):
                self.stats = sum_stats(self.stats, piece)
                continue
            self._note_first_output()
            event = self._event_for(piece)
            if event is not None:
                yield event

    def _event_for(self, piece: str | Thought) -> Event | None:
        if isinstance(piece, Thought):
            return Thinking(turn_id=self.turn_id, run_id=self.run_id, text=piece.text)
        self.text += piece
        if self.spec.output_schema is not None:
            return None
        return Token(turn_id=self.turn_id, run_id=self.run_id, text=piece)

    def _note_first_output(self) -> None:
        if self.first_output_ns is None:
            self.first_output_ns = time.perf_counter_ns()

    async def _validated_output(self, messages: list[Message]) -> AsyncIterator[Event]:
        schema = self.spec.output_schema
        self.data, problems = structured.validate_reply(self.text, schema)
        attempts = 1
        if problems:
            attempts = 2
            async for event in self._call_model(_repair_messages(messages, self.text, problems)):
                yield event
            self.data, problems = structured.validate_reply(self.text, schema)
        if problems:
            raise InvalidStructuredOutput(structured.failure_message(problems, self.text))
        yield StructuredOutput(turn_id=self.turn_id, run_id=self.run_id, data=self.data, attempts=attempts)


def system_prompt(spec: AgentSpec) -> str:
    if spec.output_schema is None:
        return spec.system_prompt
    return f"{spec.system_prompt}\n\n{structured.schema_instructions(spec.output_schema)}".strip()


def sum_stats(total: ChatStats | None, more: ChatStats | None) -> ChatStats | None:
    if total is None or more is None:
        return more or total
    summed = {field.name: getattr(total, field.name) + getattr(more, field.name)
              for field in fields(ChatStats) if field.name != "done_reason"}
    return ChatStats(**summed, done_reason=more.done_reason)


def _repair_messages(messages: list[Message], invalid_reply: str, problems: list[str]) -> list[Message]:
    return messages + [{"role": "assistant", "content": invalid_reply},
                       {"role": "user", "content": structured.repair_request(problems)}]
