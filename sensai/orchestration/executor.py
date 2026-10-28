# Agent executor: runs one AgentSpec on a list of messages and emits that run's events, from
# AgentRunStarted to AgentRunFinished. A model or output failure becomes an Error event and leaves
# the run not ok; cancellation propagates. What happens to the learner's history is the caller's call.
# An agent given tools runs a tool loop (T1): native tool calls when the model supports them (/api/show),
# otherwise a JSON {"tool", "args"} reply that is held back from the learner while it may be a call.
# An agent with strategy = "react" runs the ReAct loop of knowledge/react.py (A1) and returns data.

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing
from dataclasses import fields
from ..core.agent_spec import AgentSpec
from ..core.errors import ErrorKind
from ..core.events import (AgentRunFinished, AgentRunStarted, Error, Event, StructuredOutput, Thinking,
                           Token, ToolCall, ToolResult)
from ..core.handoff import Handoff
from ..core.ids import new_id
from ..core.messages import Message
from ..knowledge.react import ReActResult, ReActStep, observation_ok, run_react_loop
from ..knowledge.tools import ToolRegistry
from ..llm.ollama import ChatStats, OllamaError, Thought, ToolRequest
from . import structured
from .stream import stream_tokens

MAX_TOOL_RESULT_SHOWN = 2000
LAST_ROUND_NOTE = "Answer the learner now, without calling any more tools."
UNFINISHED_LOOKUP = "Sorry, I could not finish looking this up."
ROUND_SEPARATOR = "\n\n"


class InvalidStructuredOutput(Exception):
    kind = "invalid_output"


class AgentRun:

    def __init__(self, client, spec: AgentSpec, turn_id: str, parent_run_id: str | None = None,
                 tools: ToolRegistry | None = None, max_tool_rounds: int = 3):
        self.client = client
        self.spec = spec
        self.turn_id = turn_id
        self.run_id = new_id()
        self.parent_run_id = parent_run_id
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds
        self.handoff: Handoff | None = None
        self.text = ""
        self.data: dict | None = None
        self.stats: ChatStats | None = None
        self.first_output_ns: int | None = None
        self.ok = False
        self.error_kind: ErrorKind | None = None
        self.tool_calls: list[ToolRequest] = []
        self._calls: list[ToolRequest] = []
        self._round_text = ""
        self._released = ""
        self._prompted = False
        self._holding: bool | None = None
        self._final_messages: list[Message] | None = None

    async def events(self, messages: list[Message]) -> AsyncIterator[Event]:
        yield AgentRunStarted(turn_id=self.turn_id, run_id=self.run_id, agent=self.spec.name,
                              parent_run_id=self.parent_run_id, prompt_hash=self.spec.prompt_hash)
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
        if self.spec.strategy == "react":
            async for event in self._react(messages):
                yield event
            return
        if self.tools:
            async for event in self._answer_with_tools(messages):
                yield event
        else:
            async for event in self._call_model(messages):
                yield event
        if self.spec.output_schema is not None:
            async for event in self._validated_output(self._final_messages or messages):
                yield event

    async def _call_model(self, messages: list[Message], tools: list[dict] | None = None,
                          reset: bool = True) -> AsyncIterator[Event]:
        if reset:
            self.text = ""
        request_format = structured.request_format(self.spec)
        async for piece in stream_tokens(self.client, self.spec, messages, request_format, tools):
            if isinstance(piece, ChatStats):
                self.stats = sum_stats(self.stats, piece)
                continue
            self._note_first_output()
            if isinstance(piece, ToolRequest):
                self._calls.append(piece)
                continue
            event = self._event_for(piece)
            if event is not None:
                yield event

    def _event_for(self, piece: str | Thought) -> Event | None:
        if isinstance(piece, Thought):
            return Thinking(turn_id=self.turn_id, run_id=self.run_id, text=piece.text)
        self._round_text += piece
        if self._prompted:
            return self._prompted_piece()
        self.text += piece
        if self.spec.output_schema is not None:
            return None
        return Token(turn_id=self.turn_id, run_id=self.run_id, text=piece)

    def _prompted_piece(self) -> Event | None:
        if self._holding is None:
            start = self._round_text.lstrip()
            if not start or (start.startswith("`") and len(start) < 3 and set(start) == {"`"}):
                return None
            self._holding = start.startswith(("{", "```"))
            if not self._holding:
                return self._release(self._round_text)
            return None
        if self._holding:
            return None
        return self._release(self._round_text[len(self._released):])

    def _release(self, text: str) -> Event | None:
        self._released = self._round_text
        self.text += text
        if not text or self.spec.output_schema is not None:
            return None
        return Token(turn_id=self.turn_id, run_id=self.run_id, text=text)

    async def _answer_with_tools(self, messages: list[Message]) -> AsyncIterator[Event]:
        try:
            native = "tools" in await asyncio.to_thread(self.client.capabilities, self.spec.model)
        except OllamaError:
            native = False
        messages = list(messages) if native else _with_tool_instructions(messages, self.tools)
        self.text = ""
        for round_number in range(self.max_tool_rounds + 1):
            offer = round_number < self.max_tool_rounds
            if not offer:
                messages = messages + [{"role": "user", "content": LAST_ROUND_NOTE}]
            self._start_round(prompted=not native)
            async for event in self._call_model(messages, self.tools.get_native_tools() if native and offer
                                                else None, reset=False):
                yield event
            if not native:
                for event in self._end_prompted_round(accept_calls=offer):
                    yield event
            self._final_messages = messages
            if not self._calls:
                self.text = self.text.rstrip()
                return
            if (self._round_text if native else self._released).strip():
                yield self._separator()
            messages = messages + [self._assistant_turn(native)]
            for call in self._calls:
                async for event in self._run_tool(call, messages, native):
                    yield event

    def _separator(self) -> Event:
        self.text += ROUND_SEPARATOR
        return Token(turn_id=self.turn_id, run_id=self.run_id, text=ROUND_SEPARATOR)

    def _start_round(self, prompted: bool) -> None:
        self._calls = []
        self._round_text = ""
        self._released = ""
        self._prompted = prompted
        self._holding = None

    def _end_prompted_round(self, accept_calls: bool) -> Iterator[Event]:
        self._prompted = False
        if not self._holding:
            return
        call = _prompted_call(self._round_text, self.tools)
        if call is not None and accept_calls:
            self._calls.append(call)
            return
        event = self._release(UNFINISHED_LOOKUP if call is not None else self._round_text)
        if event is not None:
            yield event

    def _assistant_turn(self, native: bool) -> Message:
        if not native:
            return {"role": "assistant", "content": self._round_text}
        calls = [{"function": {"name": call.name, "arguments": call.arguments}} for call in self._calls]
        return {"role": "assistant", "content": self._round_text, "tool_calls": calls}

    async def _run_tool(self, call: ToolRequest, messages: list[Message], native: bool) -> AsyncIterator[Event]:
        self.tool_calls.append(call)
        yield ToolCall(turn_id=self.turn_id, run_id=self.run_id, name=call.name, args=call.arguments)
        result, ok = await asyncio.to_thread(self.tools.run, call.name, call.arguments)
        yield ToolResult(turn_id=self.turn_id, run_id=self.run_id, name=call.name,
                         result=result[:MAX_TOOL_RESULT_SHOWN], ok=ok)
        if native:
            messages.append({"role": "tool", "content": result, "tool_name": call.name})
        else:
            messages.append({"role": "user", "content": f"Tool result ({call.name}): {result}\n"
                                                        "Use it to answer the learner, or call another tool."})

    async def _react(self, messages: list[Message]) -> AsyncIterator[Event]:
        loop = asyncio.get_running_loop()
        steps: asyncio.Queue = asyncio.Queue()
        stopped = threading.Event()

        def report(step: ReActStep) -> None:
            loop.call_soon_threadsafe(steps.put_nowait, step)

        def work() -> ReActResult:
            try:
                return run_react_loop(self.client, self.spec.model, _last_user_text(messages),
                                      self.tools or ToolRegistry(), self.spec.max_steps, self.spec.num_ctx,
                                      report, hint=_hint(self.handoff), should_stop=stopped.is_set,
                                      system_template=self.spec.system_prompt)
            finally:
                loop.call_soon_threadsafe(steps.put_nowait, None)

        task = asyncio.ensure_future(asyncio.to_thread(work))
        try:
            while (step := await steps.get()) is not None:
                self._note_first_output()
                for event in self._react_events(step):
                    yield event
            try:
                result = await task
            except (OllamaError, InvalidStructuredOutput):
                raise
            except Exception as error:
                raise InvalidStructuredOutput(f"ReAct loop failed: {type(error).__name__}: {error}") from None
        finally:
            stopped.set()
            if not task.done():
                task.add_done_callback(_ignore_outcome)
        for stats in result.stats:
            self.stats = sum_stats(self.stats, stats)
        self.text = result.answer
        self.data = _react_data(result, self.spec, self.handoff)
        yield StructuredOutput(turn_id=self.turn_id, run_id=self.run_id, data=self.data)

    def _react_events(self, step: ReActStep) -> Iterator[Event]:
        if step.step in ("think", "verification"):
            label = "Thought" if step.step == "think" else "Verification"
            yield Thinking(turn_id=self.turn_id, run_id=self.run_id, text=f"{label}: {step.content}\n")
        elif step.step == "tool_call":
            call = ToolRequest(step.tool or "", step.args or {})
            self.tool_calls.append(call)
            yield ToolCall(turn_id=self.turn_id, run_id=self.run_id, name=call.name, args=call.arguments)
            observation = step.observation or ""
            yield ToolResult(turn_id=self.turn_id, run_id=self.run_id, name=call.name,
                             result=observation[:MAX_TOOL_RESULT_SHOWN], ok=observation_ok(observation, need_found=False))

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


def _with_tool_instructions(messages: list[Message], tools: ToolRegistry) -> list[Message]:
    instructions = (tools.get_prompt_instructions()
                    + "\nOnly when you need a tool, reply with that JSON block alone and nothing else; "
                      "otherwise answer the learner normally.")
    if messages and messages[0]["role"] == "system":
        return [{**messages[0], "content": f"{messages[0]['content']}\n\n{instructions}"}, *messages[1:]]
    return [{"role": "system", "content": instructions}, *messages]


def _prompted_call(text: str, tools: ToolRegistry) -> ToolRequest | None:
    try:
        data = json.loads(structured._strip_fences(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("tool"), str) or tools.get(data["tool"]) is None:
        return None
    args = data.get("args", {})
    return ToolRequest(data["tool"], args if isinstance(args, dict) else {})


def _last_user_text(messages: list[Message]) -> str:
    return next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")


def _hint(handoff: Handoff | None) -> str | None:
    if handoff is None or not handoff.payload:
        return None
    return json.dumps(handoff.payload, ensure_ascii=False)


def _react_data(result: ReActResult, spec: AgentSpec, handoff: Handoff | None) -> dict:
    data = {
        "sentence": result.sentence, "correction": result.correction, "rule": result.rule,
        "explanation": result.answer, "revised": result.revised, "verified_with": result.tools_used,
        "verified": result.verified,
    }
    if handoff is not None and handoff.payload:
        data["first_check"] = handoff.payload
        if not data["verified"] and handoff.payload.get("correction"):
            data["correction"] = handoff.payload["correction"]
            data["explanation"] = "Not verified with a tool; this is the first check's correction."
    if spec.handoffs:
        reason = ("explain the verified correction to the learner" if data["verified"]
                  else "the correction could not be verified; explain it with care")
        data["handoff"] = {"to": spec.handoffs[0], "reason": reason}
    return data


def _ignore_outcome(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()


def _repair_messages(messages: list[Message], invalid_reply: str, problems: list[str]) -> list[Message]:
    return messages + [{"role": "assistant", "content": invalid_reply},
                       {"role": "user", "content": structured.repair_request(problems)}]
